from __future__ import annotations

import csv
import math
import sqlite3
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


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
    Loads CSV datasets into pandas/SQLite, cleans missing or out-of-range observations,
    calculates service-level statistical distributions using NumPy/pandas, detects Z-score
    outliers, and generates Matplotlib SVG/PNG visualisations.
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
                        or math.isnan(rec.latency_ms)
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

    def to_dataframe(self) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM service_telemetry ORDER BY record_id ASC",
            self._conn,
        )

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
        df = self.to_dataframe()
        if df.empty:
            return []

        results: list[ServiceStatistics] = []
        for svc, group in df.groupby("service", sort=True):
            lat = group["latency_ms"].to_numpy(dtype=float)
            err = group["error_rate_pct"].to_numpy(dtype=float)
            rps = int(group["requests_per_sec"].sum())

            std_dev = float(np.std(lat, ddof=0)) if len(lat) > 1 else 0.0
            results.append(
                ServiceStatistics(
                    service=str(svc),
                    sample_count=len(lat),
                    mean_latency_ms=round(float(np.mean(lat)), 2),
                    median_latency_ms=round(float(np.median(lat)), 2),
                    p95_latency_ms=round(float(np.percentile(lat, 95)), 2),
                    std_dev_latency_ms=round(std_dev, 2),
                    mean_error_rate_pct=round(float(np.mean(err)), 3),
                    total_requests_per_sec=rps,
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

    def export_latency_chart(self, output_path: Path) -> Path:
        """Generates a dark-themed Matplotlib bar & p95 comparison chart from the loaded dataset."""
        summaries = self.summarize_by_service()
        if not summaries:
            raise ValueError("Cannot generate chart from empty dataset")

        services = [s.service for s in summaries]
        medians = [s.median_latency_ms for s in summaries]
        p95s = [s.p95_latency_ms for s in summaries]

        fig, ax = plt.subplots(figsize=(8, 4.2), dpi=140)
        fig.patch.set_facecolor("#0d0f12")
        ax.set_facecolor("#13161b")

        x = np.arange(len(services))
        width = 0.36
        ax.bar(x - width / 2, medians, width, label="Median (p50) ms", color="#38bdf8")
        ax.bar(x + width / 2, p95s, width, label="Tail (p95) ms", color="#f59e0b")

        ax.set_title("Service Response Latency: Median vs 95th Percentile (ms)", color="#e5e7eb", pad=12)
        ax.set_ylabel("Latency (ms)", color="#9ca3af")
        ax.set_xticks(x)
        ax.set_xticklabels(services, color="#e5e7eb")
        ax.tick_params(axis="y", colors="#9ca3af")
        for spine in ax.spines.values():
            spine.set_color("#262b35")
        ax.legend(facecolor="#13161b", edgecolor="#262b35", labelcolor="#e5e7eb")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, facecolor=fig.get_facecolor())
        plt.close(fig)
        return output_path


def pearson_correlation(xs: list[float], ys: list[float]) -> float:
    """Computes Pearson correlation coefficient r in [-1.0, 1.0]."""
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("Input series must have identical length >= 2")
    arr_x = np.asarray(xs, dtype=float)
    arr_y = np.asarray(ys, dtype=float)
    if np.std(arr_x) == 0.0 or np.std(arr_y) == 0.0:
        return 0.0
    corr = float(np.corrcoef(arr_x, arr_y)[0, 1])
    return round(corr, 4)
