# Performance Cost Regression for Firefox-Blocked Tracker Requests

Predicting the transfer size of tracker requests blocked by Firefox Enhanced Tracking Protection, using pre-response features available at block time.

## Problem

When Firefox blocks a tracker request, the response is never observed. The browser sees the URL, resource type, and request metadata, but not the response size. This project trains models to predict what the response would have been, enabling the privacy dashboard to show users concrete savings: "Firefox saved you approximately 2.3MB of bandwidth this week."

A domain-level lookup table is insufficient because the same domain serves vastly different resources (`googletagmanager.com/gtag/js` = 93KB script, `googletagmanager.com/collect` = 0-byte beacon). A path-level table is infeasible (50M entries, 1.1GB, immediately stale).

## Results

- **XGBoost with Tweedie loss**: 39.4% MAE improvement over lookup table baseline
- **Character-level URL CNN**: best ranking quality (Spearman 0.977)
- **Weekly aggregate accuracy**: within 10% of true total 63% of the time (vs 15% for LUT)
- 10 model architectures compared; loss function selection matters more than architecture

## Structure

```
sql/                    BigQuery queries for HTTP Archive data extraction
src/model/
  train_per_request.py  Main pipeline: LUT, Ridge, RF, XGBoost, LightGBM, CatBoost
  train_advanced.py     Tweedie loss, two-stage, URL hashing, quantile regression
  advanced_analysis.py  Calibration, aggregation simulation, distribution shift
  url_cnn.py            Character-level CNN for learned URL representations
data/
  raw/                  HTTP Archive request data
  external/             Disconnect list
models/per_request/     Trained models and results
output/                 Visualizations
```

## Data

Training data: 348,909 tracker requests from the HTTP Archive June 2024 crawl (0.1% sample of 348M total). Features: URL path structure, file extension, resource type, initiator type, target-encoded domain statistics. Target: `transfer_bytes`.

## Paper

`performance_cost_estimation_tracker_domains.tex` (18 pages, compile with `pdflatex` + `bibtex`)
