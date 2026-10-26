from __future__ import annotations

import csv
import math
import sqlite3
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TelemetryRecord:
    record_id: str
    service: str
    region: str
    latency_ms: float
    error_rate_pct: float
    requests_per_sec: int


@dataclass(frozen=True)
class ServiceStatistics:
    service: str
    sample_count: int
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    std_dev_latency_ms: float
    mean_error_rate_pct: float
    total_requests_per_sec: int


class AnalyticsWorkbench:
    """
    Ingests raw CSV or programmatic telemetry records, validates numerical bounds,
    stores clean rows in an indexed SQLite analytical table, and computes cohort
    aggregations, Pearson correlations, and Z-score latency anomalies.
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS service_telemetry (
                record_id TEXT PRIMARY KEY,
                service TEXT NOT NULL,
                region TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                error_rate_pct REAL NOT NULL,
                requests_per_sec INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_telemetry_service
                ON service_telemetry(service);
            """
        )
        self._conn.commit()

    def ingest_csv(self, csv_path: Path) -> dict[str, int]:
        inserted = 0
        rejected = 0

        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    rec = TelemetryRecord(
                        record_id=(row.get("record_id") or "").strip(),
                        service=(row.get("service") or "").strip(),
                        region=(row.get("region") or "").strip(),
                        latency_ms=float(row.get("latency_ms") or ""),
                        error_rate_pct=float(row.get("error_rate_pct") or ""),
                        requests_per_sec=int(row.get("requests_per_sec") or ""),
                    )
                    if (
                        not rec.record_id
                        or not rec.service
                        or not rec.region
                        or rec.latency_ms < 0.0
                        or not (0.0 <= rec.error_rate_pct <= 100.0)
                        or rec.requests_per_sec < 0
                    ):
                        rejected += 1
                        continue
                    self._insert_record(rec)
                    inserted += 1
                except (ValueError, TypeError):
                    rejected += 1

        self._conn.commit()
        return {"inserted": inserted, "rejected": rejected}

    def _insert_record(self, rec: TelemetryRecord) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO service_telemetry (
                record_id, service, region, latency_ms, error_rate_pct, requests_per_sec
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                rec.record_id,
                rec.service,
                rec.region,
                rec.latency_ms,
                rec.error_rate_pct,
                rec.requests_per_sec,
            ),
        )

    def summarize_by_service(self) -> list[ServiceStatistics]:
        services = [
            r["service"]
            for r in self._conn.execute(
                "SELECT DISTINCT service FROM service_telemetry ORDER BY service ASC"
            ).fetchall()
        ]
        results: list[ServiceStatistics] = []

        for svc in services:
            rows = self._conn.execute(
                """
                SELECT latency_ms, error_rate_pct, requests_per_sec
                FROM service_telemetry
                WHERE service = ?
                ORDER BY latency_ms ASC
                """,
                (svc,),
            ).fetchall()

            latencies = [float(r["latency_ms"]) for r in rows]
            errors = [float(r["error_rate_pct"]) for r in rows]
            rps_total = sum(int(r["requests_per_sec"]) for r in rows)

            std_dev = statistics.pstdev(latencies) if len(latencies) > 1 else 0.0
            results.append(
                ServiceStatistics(
                    service=svc,
                    sample_count=len(latencies),
                    mean_latency_ms=round(statistics.fmean(latencies), 2),
                    median_latency_ms=round(statistics.median(latencies), 2),
                    p95_latency_ms=round(_quantile(latencies, 0.95), 2),
                    std_dev_latency_ms=round(std_dev, 2),
                    mean_error_rate_pct=round(statistics.fmean(errors), 3),
                    total_requests_per_sec=rps_total,
                )
            )
        return results

    def detect_latency_anomalies(
        self,
        service: str,
        z_threshold: float = 2.0,
    ) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT record_id, region, latency_ms
            FROM service_telemetry
            WHERE service = ?
            ORDER BY record_id ASC
            """,
            (service,),
        ).fetchall()

        if len(rows) < 2:
            return []

        latencies = [float(r["latency_ms"]) for r in rows]
        mean = statistics.fmean(latencies)
        std = statistics.pstdev(latencies)
        if std == 0.0:
            return []

        anomalies: list[dict[str, Any]] = []
        for r in rows:
            val = float(r["latency_ms"])
            z_score = (val - mean) / std
            if abs(z_score) >= z_threshold:
                anomalies.append(
                    {
                        "record_id": r["record_id"],
                        "region": r["region"],
                        "latency_ms": val,
                        "z_score": round(z_score, 2),
                    }
                )
        return anomalies


def pearson_correlation(xs: list[float], ys: list[float]) -> float:
    """Computes Pearson correlation coefficient r in [-1.0, 1.0]."""
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("Input series must have identical length >= 2")
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return 0.0
    return round(num / (den_x * den_y), 4)


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = q * (len(sorted_vals) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_vals[lo]
    weight = idx - lo
    return sorted_vals[lo] * (1.0 - weight) + sorted_vals[hi] * weight
