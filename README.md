# Data Analysis Workbench

A reproducible data ingestion, SQLite analytical query engine, cohort aggregation, and statistical anomaly detection workbench built in **Python 3.12**.

## Overview

`data-analysis-workbench` ingests operational CSV datasets (labelled synthetic or production service telemetry), validates domain bounds (non-negative latencies, bounded error percentages, non-empty primary keys), loads clean records into an indexed SQLite table, and produces service-level statistical summaries (`mean`, `median`, `p95`, population standard deviation), Pearson correlation coefficients, and Z-score anomaly reports.

## Key Capabilities

- **Schema & Domain Validation:** Rejects malformed or out-of-range observations before analytical storage and reports exact `inserted` vs. `rejected` counts.
- **SQL + Statistical Aggregation:** Combines indexed SQLite queries with exact quantile interpolation (`p95`), `statistics.fmean`, `statistics.median`, and `statistics.pstdev`.
- **Z-Score Outlier Detection:** Flags records whose latency deviates by $\ge 2.0\sigma$ from the service baseline.
- **Bivariate Correlation Analysis:** Computes Pearson's $r$ between request throughput and response latency.

## Running Tests & Linter

```bash
ruff check .
pytest -v
```
