"""
Correlated-browsing aggregation robustness check.

Tests whether the model's aggregation advantage holds when
browsing is correlated (users visit the same domains repeatedly)
rather than sampling uniformly.

Three strategies:
  1. Uniform: sample N requests randomly (existing approach)
  2. Domain-correlated: sample K domains, then N requests from those domains
  3. Page-correlated: sample P pages, include all tracker requests from each

Usage:
  python src/model/correlated_aggregation.py
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from pathlib import Path
import json
import time
import sys

sys.path.insert(0, str(Path(__file__).parent))
from url_embeddings import URLEmbedder
from train_multi_target import engineer_features

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models" / "per_request"


def uniform_aggregation(y_true, y_pred, N, n_trials=2000):
    """Standard uniform sampling."""
    errs = []
    for _ in range(n_trials):
        idx = np.random.choice(len(y_true), size=N, replace=True)
        true_sum = y_true[idx].sum()
        if true_sum == 0:
            continue
        pred_sum = y_pred[idx].sum()
        errs.append(abs(pred_sum - true_sum) / true_sum * 100)
    return errs


def domain_correlated_aggregation(y_true, y_pred, domains, N,
                                   n_domains=15, n_trials=2000):
    """Sample K domains, then sample N total requests from those domains."""
    unique_domains = domains.unique()
    domain_indices = {d: np.where(domains.values == d)[0] for d in unique_domains}

    errs = []
    for _ in range(n_trials):
        # Pick n_domains domains (with replacement, simulating repeat visits)
        chosen = np.random.choice(unique_domains, size=n_domains, replace=True)

        # Collect all candidate indices from chosen domains
        pool = np.concatenate([domain_indices[d] for d in chosen])

        if len(pool) < N:
            # If not enough requests, use all of them
            idx = pool
        else:
            idx = np.random.choice(pool, size=N, replace=False)

        true_sum = y_true[idx].sum()
        if true_sum == 0:
            continue
        pred_sum = y_pred[idx].sum()
        errs.append(abs(pred_sum - true_sum) / true_sum * 100)
    return errs


def page_correlated_aggregation(y_true, y_pred, pages, N,
                                 n_pages=30, n_trials=1000):
    """Sample P pages, include all tracker requests from each page."""
    # Pre-filter to pages with enough requests to be meaningful
    page_counts = pages.value_counts()
    viable_pages = page_counts[page_counts >= 2].index.values
    page_indices = {p: np.where(pages.values == p)[0] for p in viable_pages}
    unique_pages = viable_pages

    errs = []
    for _ in range(n_trials):
        # Pick pages until we have ~N requests
        chosen_pages = np.random.choice(unique_pages, size=n_pages, replace=True)
        idx = np.concatenate([page_indices[p] for p in chosen_pages])

        # Subsample if too many
        if len(idx) > N * 2:
            idx = np.random.choice(idx, size=N, replace=False)

        if len(idx) == 0:
            continue
        true_sum = y_true[idx].sum()
        if true_sum == 0:
            continue
        pred_sum = y_pred[idx].sum()
        errs.append(abs(pred_sum - true_sum) / true_sum * 100)
    return errs


def main():
    np.random.seed(42)

    # Load data
    print("Loading data...")
    df = pd.read_csv(ROOT / 'data' / 'raw' / 'per_request_1pct.csv', low_memory=False)
    n = len(df)
    idx = np.random.permutation(n)
    train_df = df.iloc[idx[:int(0.7 * n)]].reset_index(drop=True)
    test_df = df.iloc[idx[int(0.85 * n):]].reset_index(drop=True)

    # Embeddings and features
    print("Building features...")
    embedder = URLEmbedder(n_components=50)
    embedder.fit(train_df['url_path'].fillna(''))
    embed_test = embedder.transform(test_df['url_path'].fillna(''))
    X_test = engineer_features(test_df, train_df, embed_test)

    y_test = test_df['transfer_bytes'].clip(lower=0).values
    domains = test_df['tracker_domain']
    pages = test_df['page_domain']

    print(f"Test set: {len(test_df):,} rows, {domains.nunique():,} domains, {pages.nunique():,} pages")

    # LUT predictions (vectorized)
    g = train_df['transfer_bytes'].median()
    dt_med = train_df.groupby(['tracker_domain', 'resource_type'])['transfer_bytes'].median()
    d_med = train_df.groupby('tracker_domain')['transfer_bytes'].median()
    dt_df = dt_med.rename('_pred').reset_index()
    lut_preds = test_df[['tracker_domain', 'resource_type']].merge(
        dt_df, on=['tracker_domain', 'resource_type'], how='left')['_pred']
    lut_preds = lut_preds.fillna(test_df['tracker_domain'].map(d_med))
    lut_preds = lut_preds.fillna(g).values

    # Model predictions
    model = xgb.XGBRegressor()
    model.load_model(str(MODELS / 'xgb_transfer_bytes.json'))
    model_preds = np.clip(model.predict(X_test) - 1, 0, None)

    # Run all three strategies
    results = {}
    for N in [50, 100, 200, 500]:
        print(f"\nN={N}:")
        row = {}

        # Uniform
        t0 = time.time()
        model_errs = uniform_aggregation(y_test, model_preds, N)
        lut_errs = uniform_aggregation(y_test, lut_preds, N)
        row['uniform'] = {
            'model_median': float(np.median(model_errs)),
            'lut_median': float(np.median(lut_errs)),
            'model_within_10': float(np.mean(np.array(model_errs) < 10) * 100),
            'lut_within_10': float(np.mean(np.array(lut_errs) < 10) * 100),
        }
        print(f"  Uniform:    Model {row['uniform']['model_median']:.1f}%, "
              f"LUT {row['uniform']['lut_median']:.1f}%  ({time.time()-t0:.1f}s)")

        # Domain-correlated
        t0 = time.time()
        model_errs = domain_correlated_aggregation(y_test, model_preds, domains, N, n_domains=15)
        lut_errs = domain_correlated_aggregation(y_test, lut_preds, domains, N, n_domains=15)
        row['domain_corr'] = {
            'model_median': float(np.median(model_errs)),
            'lut_median': float(np.median(lut_errs)),
            'model_within_10': float(np.mean(np.array(model_errs) < 10) * 100),
            'lut_within_10': float(np.mean(np.array(lut_errs) < 10) * 100),
        }
        print(f"  Domain-15:  Model {row['domain_corr']['model_median']:.1f}%, "
              f"LUT {row['domain_corr']['lut_median']:.1f}%  ({time.time()-t0:.1f}s)")

        # Page-correlated
        t0 = time.time()
        model_errs = page_correlated_aggregation(y_test, model_preds, pages, N, n_pages=30)
        lut_errs = page_correlated_aggregation(y_test, lut_preds, pages, N, n_pages=30)
        row['page_corr'] = {
            'model_median': float(np.median(model_errs)),
            'lut_median': float(np.median(lut_errs)),
            'model_within_10': float(np.mean(np.array(model_errs) < 10) * 100),
            'lut_within_10': float(np.mean(np.array(lut_errs) < 10) * 100),
        }
        print(f"  Page-30:    Model {row['page_corr']['model_median']:.1f}%, "
              f"LUT {row['page_corr']['lut_median']:.1f}%  ({time.time()-t0:.1f}s)")

        results[N] = row

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY: Median weekly error (%)")
    print(f"{'='*70}")
    print(f"{'N':>5s}  {'Uniform':>18s}  {'Domain-corr (15)':>18s}  {'Page-corr (30)':>18s}")
    print(f"{'':>5s}  {'Model':>8s} {'LUT':>8s}  {'Model':>8s} {'LUT':>8s}  {'Model':>8s} {'LUT':>8s}")
    print("-" * 70)
    for N in [50, 100, 200, 500]:
        r = results[N]
        print(f"{N:5d}  {r['uniform']['model_median']:>7.1f}% {r['uniform']['lut_median']:>7.1f}%  "
              f"{r['domain_corr']['model_median']:>7.1f}% {r['domain_corr']['lut_median']:>7.1f}%  "
              f"{r['page_corr']['model_median']:>7.1f}% {r['page_corr']['lut_median']:>7.1f}%")

    # Save
    with open(MODELS / 'correlated_aggregation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {MODELS / 'correlated_aggregation_results.json'}")


if __name__ == '__main__':
    main()
