from __future__ import annotations

from pathlib import Path

from src.data_workbench.pipeline import AnalyticsWorkbench, pearson_correlation


def test_csv_ingestion_filters_malformed_rows_and_computes_cohort_stats(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "telemetry.csv"
    csv_path.write_text(
        "record_id,service,region,latency_ms,error_rate_pct,requests_per_sec\n"
        "r1,checkout-api,eu-west-1,24.0,0.1,420\n"
        "r2,checkout-api,eu-west-1,26.0,0.1,450\n"
        "r3,checkout-api,eu-west-2,25.0,0.2,410\n"
        "r4,checkout-api,eu-west-2,24.5,0.1,430\n"
        "r5,checkout-api,us-east-1,25.5,0.1,440\n"
        "r6,checkout-api,us-east-1,185.0,2.4,610\n"
        "bad1,checkout-api,eu-west-1,-10.0,0.1,100\n"
        "bad2,checkout-api,eu-west-1,20.0,150.0,100\n",
        encoding="utf-8",
    )

    wb = AnalyticsWorkbench(":memory:")
    counts = wb.ingest_csv(csv_path)
    assert counts["inserted"] == 6
    assert counts["rejected"] == 2

    summaries = wb.summarize_by_service()
    assert len(summaries) == 1
    svc = summaries[0]
    assert svc.service == "checkout-api"
    assert svc.sample_count == 6
    assert svc.median_latency_ms == 25.25
    assert svc.total_requests_per_sec == 2760

    anomalies = wb.detect_latency_anomalies("checkout-api", z_threshold=2.0)
    assert len(anomalies) == 1
    assert anomalies[0]["record_id"] == "r6"
    assert anomalies[0]["z_score"] > 2.0


def test_pearson_correlation_matches_linear_relationship() -> None:
    rps = [100.0, 200.0, 300.0, 400.0, 500.0]
    latency = [12.0, 24.0, 36.0, 48.0, 60.0]
    assert pearson_correlation(rps, latency) == 1.0
