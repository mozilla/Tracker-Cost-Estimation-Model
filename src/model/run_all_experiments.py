"""
Comprehensive experiments for the paper.

Runs all baselines, ablations, and analysis needed for a rigorous submission:
  1. Multiple baselines: global median, domain LUT, domain+type LUT, path LUT, hybrid path+model
  2. Loss function ablation on transfer_bytes
  3. Feature ablation: with/without TF-IDF embeddings, with/without regex features
  4. Aggregation accuracy for all approaches
  5. Bootstrap confidence intervals on all metrics
  6. Per-resource-type breakdown

Usage:
  python src/model/run_all_experiments.py
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, median_absolute_error
from scipy.stats import spearmanr
from pathlib import Path
import json
import time
import sys

sys.path.insert(0, str(Path(__file__).parent))
from url_embeddings import URLEmbedder
from train_multi_target import engineer_features

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models" / "per_request"
MODELS.mkdir(parents=True, exist_ok=True)


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, ci=95):
    """Bootstrap confidence interval for a metric."""
    np.random.seed(42)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    lo = np.percentile(scores, (100 - ci) / 2)
    hi = np.percentile(scores, 100 - (100 - ci) / 2)
    return np.mean(scores), lo, hi


def aggregation_sim(y_true, y_pred, Ns=[50, 100, 200, 500], n_trials=2000):
    """Aggregation accuracy simulation."""
    results = {}
    for N in Ns:
        errs = []
        for _ in range(n_trials):
            idx = np.random.choice(len(y_true), size=N, replace=True)
            true_sum = y_true[idx].sum()
            if true_sum == 0:
                continue
            pred_sum = y_pred[idx].sum()
            errs.append(abs(pred_sum - true_sum) / true_sum * 100)
        results[N] = {
            'median_pct_err': float(np.median(errs)),
            'within_10pct': float(np.mean(np.array(errs) < 10) * 100),
            'within_5pct': float(np.mean(np.array(errs) < 5) * 100),
            'p25_err': float(np.percentile(errs, 25)),
            'p75_err': float(np.percentile(errs, 75)),
        }
    return results


def path_lut_predict(train_df, test_df, target_col):
    """Path-level LUT with fallback chain: path -> domain+type -> domain -> global."""
    path_med = train_df.groupby(['tracker_domain', 'url_path'])[target_col].median()
    dt_med = train_df.groupby(['tracker_domain', 'resource_type'])[target_col].median()
    d_med = train_df.groupby('tracker_domain')[target_col].median()
    g_med = train_df[target_col].median()

    preds = []
    path_hits = 0
    for _, row in test_df.iterrows():
        pk = (row['tracker_domain'], row['url_path'])
        dtk = (row['tracker_domain'], row['resource_type'])
        dk = row['tracker_domain']

        if pk in path_med.index:
            preds.append(path_med[pk])
            path_hits += 1
        elif dtk in dt_med.index:
            preds.append(dt_med[dtk])
        elif dk in d_med.index:
            preds.append(d_med[dk])
        else:
            preds.append(g_med)

    return np.array(preds), path_hits / len(test_df)


def path_lut_predict_fast(train_df, test_df, target_col):
    """Vectorized path LUT."""
    path_med = train_df.groupby(['tracker_domain', 'url_path'])[target_col].median()
    dt_med = train_df.groupby(['tracker_domain', 'resource_type'])[target_col].median()
    d_med = train_df.groupby('tracker_domain')[target_col].median()
    g_med = train_df[target_col].median()

    # Path lookup
    path_keys = pd.MultiIndex.from_frame(test_df[['tracker_domain', 'url_path']])
    preds = pd.Series([path_med.get(k, np.nan) for k in path_keys], index=test_df.index)
    path_hit = preds.notna()

    # Fallback: domain+type
    mask = preds.isna()
    dt_keys = pd.MultiIndex.from_frame(test_df.loc[mask, ['tracker_domain', 'resource_type']])
    preds[mask] = [dt_med.get(k, np.nan) for k in dt_keys]

    # Fallback: domain
    mask = preds.isna()
    preds[mask] = test_df.loc[mask, 'tracker_domain'].map(d_med)

    # Fallback: global
    preds = preds.fillna(g_med)

    return preds.values, path_hit.mean()


def domain_type_lut_predict(train_df, test_df, target_col):
    """Domain+type LUT with fallback."""
    dt_med = train_df.groupby(['tracker_domain', 'resource_type'])[target_col].median()
    d_med = train_df.groupby('tracker_domain')[target_col].median()
    g_med = train_df[target_col].median()

    dt_keys = pd.MultiIndex.from_frame(test_df[['tracker_domain', 'resource_type']])
    preds = pd.Series([dt_med.get(k, np.nan) for k in dt_keys], index=test_df.index)

    mask = preds.isna()
    preds[mask] = test_df.loc[mask, 'tracker_domain'].map(d_med)
    preds = preds.fillna(g_med)

    return preds.values


def hybrid_predict(path_preds, path_hit_mask, model_preds):
    """Use path LUT when available, model otherwise."""
    preds = model_preds.copy()
    preds[path_hit_mask] = path_preds[path_hit_mask]
    return preds


def main():
    np.random.seed(42)

    # Load data
    print("Loading data...")
    t0 = time.time()
    df = pd.read_csv(ROOT / 'data' / 'raw' / 'per_request_1pct.csv', low_memory=False)
    print(f"Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    # Split
    n = len(df)
    idx = np.random.permutation(n)
    train_df = df.iloc[idx[:int(0.7 * n)]].reset_index(drop=True)
    val_df = df.iloc[idx[int(0.7 * n):int(0.85 * n)]].reset_index(drop=True)
    test_df = df.iloc[idx[int(0.85 * n):]].reset_index(drop=True)
    print(f"Split: train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")

    # URL embeddings
    print("\nFitting URL embeddings...")
    embedder = URLEmbedder(n_components=50)
    embedder.fit(train_df['url_path'].fillna(''))
    embed_train = embedder.transform(train_df['url_path'].fillna(''))
    embed_val = embedder.transform(val_df['url_path'].fillna(''))
    embed_test = embedder.transform(test_df['url_path'].fillna(''))

    # Features
    X_train = engineer_features(train_df, train_df, embed_train)
    X_val = engineer_features(val_df, train_df, embed_val)
    X_test = engineer_features(test_df, train_df, embed_test)

    # Also build features WITHOUT embeddings for ablation
    X_train_no_emb = engineer_features(train_df, train_df, None)
    X_val_no_emb = engineer_features(val_df, train_df, None)
    X_test_no_emb = engineer_features(test_df, train_df, None)

    # Also build features WITHOUT regex for ablation
    regex_cols = [c for c in X_train.columns if c.startswith('path_has_')]
    X_train_no_regex = X_train.drop(columns=regex_cols)
    X_val_no_regex = X_val.drop(columns=regex_cols)
    X_test_no_regex = X_test.drop(columns=regex_cols)

    target_col = 'transfer_bytes'
    y_train = train_df[target_col].clip(lower=0)
    y_val = val_df[target_col].clip(lower=0)
    y_test = test_df[target_col].clip(lower=0)

    all_results = {}

    # ============================================================
    # 1. BASELINES
    # ============================================================
    print("\n" + "=" * 70)
    print("BASELINES")
    print("=" * 70)

    # Global median
    g_med = train_df[target_col].median()
    global_preds = np.full(len(y_test), g_med)
    mae = mean_absolute_error(y_test, global_preds)
    print(f"Global median:    MAE={mae:,.0f}")
    all_results['global_median'] = {'mae': float(mae), 'preds': global_preds}

    # Domain-only LUT
    d_med = train_df.groupby('tracker_domain')[target_col].median()
    domain_preds = test_df['tracker_domain'].map(d_med).fillna(g_med).values
    mae = mean_absolute_error(y_test, domain_preds)
    print(f"Domain LUT:       MAE={mae:,.0f}")
    all_results['domain_lut'] = {'mae': float(mae), 'preds': domain_preds}

    # Domain + type LUT
    dt_preds = domain_type_lut_predict(train_df, test_df, target_col)
    mae = mean_absolute_error(y_test, dt_preds)
    print(f"Domain+type LUT:  MAE={mae:,.0f}")
    all_results['domain_type_lut'] = {'mae': float(mae), 'preds': dt_preds}

    # Path LUT with fallback
    print("Computing path LUT (this may take a minute)...")
    path_preds, path_coverage = path_lut_predict_fast(train_df, test_df, target_col)
    mae = mean_absolute_error(y_test, path_preds)
    print(f"Path LUT:         MAE={mae:,.0f} (coverage={path_coverage*100:.1f}%)")
    all_results['path_lut'] = {'mae': float(mae), 'preds': path_preds,
                               'coverage': float(path_coverage)}

    # ============================================================
    # 2. LOSS FUNCTION ABLATION
    # ============================================================
    print("\n" + "=" * 70)
    print("LOSS FUNCTION ABLATION (transfer_bytes)")
    print("=" * 70)

    base_params = dict(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
        tree_method='hist', random_state=42, verbosity=0,
        n_jobs=-1, early_stopping_rounds=20,
    )

    loss_configs = {
        'squared_error': {'objective': 'reg:squarederror'},
        'huber': {'objective': 'reg:pseudohubererror', 'huber_slope': 10000},
        'tweedie_1.2': {'objective': 'reg:tweedie', 'tweedie_variance_power': 1.2},
        'tweedie_1.5': {'objective': 'reg:tweedie', 'tweedie_variance_power': 1.5},
        'tweedie_1.8': {'objective': 'reg:tweedie', 'tweedie_variance_power': 1.8},
    }

    for name, loss_params in loss_configs.items():
        params = {**base_params, **loss_params}
        offset = 1 if 'tweedie' in name else 0
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train + offset,
                  eval_set=[(X_val, y_val + offset)], verbose=False)
        preds = np.clip(model.predict(X_test) - offset, 0, None)
        mae = mean_absolute_error(y_test, preds)
        rho = spearmanr(y_test, preds).statistic
        best_iter = getattr(model, 'best_iteration', 500)
        print(f"{name:<18s} MAE={mae:>8,.0f}  rho={rho:.4f}  iter={best_iter}")
        all_results[f'loss_{name}'] = {
            'mae': float(mae), 'rho': float(rho), 'preds': preds,
            'best_iteration': int(best_iter),
        }

    # ============================================================
    # 3. FEATURE ABLATION
    # ============================================================
    print("\n" + "=" * 70)
    print("FEATURE ABLATION (transfer_bytes, Tweedie p=1.5)")
    print("=" * 70)

    tweedie_params = {**base_params, 'objective': 'reg:tweedie',
                      'tweedie_variance_power': 1.5}

    # Full features (TF-IDF + regex + all)
    model_full = xgb.XGBRegressor(**tweedie_params)
    model_full.fit(X_train, y_train + 1,
                   eval_set=[(X_val, y_val + 1)], verbose=False)
    preds_full = np.clip(model_full.predict(X_test) - 1, 0, None)
    mae_full = mean_absolute_error(y_test, preds_full)
    print(f"Full (TF-IDF + regex):    MAE={mae_full:,.0f}")
    all_results['feat_full'] = {'mae': float(mae_full), 'preds': preds_full}

    # No embeddings (regex only)
    model_no_emb = xgb.XGBRegressor(**tweedie_params)
    model_no_emb.fit(X_train_no_emb, y_train + 1,
                     eval_set=[(X_val_no_emb, y_val + 1)], verbose=False)
    preds_no_emb = np.clip(model_no_emb.predict(X_test_no_emb) - 1, 0, None)
    mae_no_emb = mean_absolute_error(y_test, preds_no_emb)
    print(f"No TF-IDF (regex only):   MAE={mae_no_emb:,.0f}")
    all_results['feat_no_emb'] = {'mae': float(mae_no_emb), 'preds': preds_no_emb}

    # No regex (TF-IDF only)
    model_no_regex = xgb.XGBRegressor(**tweedie_params)
    model_no_regex.fit(X_train_no_regex, y_train + 1,
                       eval_set=[(X_val_no_regex, y_val + 1)], verbose=False)
    preds_no_regex = np.clip(model_no_regex.predict(X_test_no_regex) - 1, 0, None)
    mae_no_regex = mean_absolute_error(y_test, preds_no_regex)
    print(f"No regex (TF-IDF only):   MAE={mae_no_regex:,.0f}")
    all_results['feat_no_regex'] = {'mae': float(mae_no_regex), 'preds': preds_no_regex}

    # ============================================================
    # 4. HYBRID: PATH LUT + MODEL
    # ============================================================
    print("\n" + "=" * 70)
    print("HYBRID: PATH LUT + MODEL FALLBACK")
    print("=" * 70)

    # Identify which test rows the path LUT covers
    path_med = train_df.groupby(['tracker_domain', 'url_path'])[target_col].median()
    path_keys = pd.MultiIndex.from_frame(test_df[['tracker_domain', 'url_path']])
    path_hit_mask = np.array([k in path_med.index for k in path_keys])

    hybrid_preds = hybrid_predict(path_preds, path_hit_mask, preds_full)
    mae_hybrid = mean_absolute_error(y_test, hybrid_preds)
    print(f"Hybrid (path LUT + model): MAE={mae_hybrid:,.0f}")
    print(f"  Path LUT covers {path_hit_mask.mean()*100:.1f}% of test rows")
    print(f"  Path LUT MAE on covered: {mean_absolute_error(y_test[path_hit_mask], path_preds[path_hit_mask]):,.0f}")
    print(f"  Model MAE on uncovered:  {mean_absolute_error(y_test[~path_hit_mask], preds_full[~path_hit_mask]):,.0f}")
    all_results['hybrid'] = {'mae': float(mae_hybrid), 'preds': hybrid_preds,
                             'path_coverage': float(path_hit_mask.mean())}

    # ============================================================
    # 5. BOOTSTRAP CONFIDENCE INTERVALS
    # ============================================================
    print("\n" + "=" * 70)
    print("BOOTSTRAP 95% CIs (1000 resamples)")
    print("=" * 70)

    approaches = {
        'Domain+type LUT': dt_preds,
        'Path LUT': path_preds,
        'XGB Tweedie (full)': preds_full,
        'Hybrid': hybrid_preds,
    }

    y_test_arr = y_test.values
    for name, preds in approaches.items():
        mean_mae, lo, hi = bootstrap_ci(y_test_arr, preds, mean_absolute_error)
        rho = spearmanr(y_test_arr, preds).statistic
        print(f"{name:<25s} MAE={mean_mae:>8,.0f} [{lo:>8,.0f}, {hi:>8,.0f}]  rho={rho:.4f}")
        all_results[f'ci_{name}'] = {'mae': float(mean_mae), 'ci_lo': float(lo),
                                     'ci_hi': float(hi), 'rho': float(rho)}

    # ============================================================
    # 6. AGGREGATION ACCURACY
    # ============================================================
    print("\n" + "=" * 70)
    print("AGGREGATION ACCURACY (transfer_bytes)")
    print("=" * 70)

    for name, preds in approaches.items():
        agg = aggregation_sim(y_test_arr, preds)
        print(f"\n{name}:")
        print(f"  {'N':>5s}  {'Med err':>8s}  {'<10%':>8s}  {'<5%':>8s}")
        for N, r in agg.items():
            print(f"  {N:5d}  {r['median_pct_err']:>7.1f}%  {r['within_10pct']:>7.1f}%  {r['within_5pct']:>7.1f}%")
        all_results[f'agg_{name}'] = agg

    # ============================================================
    # 7. PER RESOURCE TYPE
    # ============================================================
    print("\n" + "=" * 70)
    print("PER RESOURCE TYPE (transfer_bytes, XGB Tweedie full)")
    print("=" * 70)

    print(f"{'Type':<10s} {'n':>8s} {'LUT MAE':>10s} {'Model MAE':>10s} {'Hybrid MAE':>10s} {'Model vs LUT':>12s}")
    print("-" * 62)
    rt_results = {}
    for rt in ['script', 'image', 'other', 'html', 'text', 'css']:
        mask = test_df['resource_type'] == rt
        if mask.sum() < 100:
            continue
        yt = y_test[mask].values
        lut_m = mean_absolute_error(yt, dt_preds[mask])
        mod_m = mean_absolute_error(yt, preds_full[mask])
        hyb_m = mean_absolute_error(yt, hybrid_preds[mask])
        improv = (1 - mod_m / lut_m) * 100
        print(f"{rt:<10s} {mask.sum():>8,d} {lut_m:>10,.0f} {mod_m:>10,.0f} {hyb_m:>10,.0f} {improv:>+11.1f}%")
        rt_results[rt] = {
            'n': int(mask.sum()), 'lut_mae': float(lut_m),
            'model_mae': float(mod_m), 'hybrid_mae': float(hyb_m),
        }
    all_results['per_resource_type'] = rt_results

    # ============================================================
    # SAVE
    # ============================================================
    # Remove numpy arrays before saving
    save_results = {}
    for k, v in all_results.items():
        if isinstance(v, dict):
            save_results[k] = {kk: vv for kk, vv in v.items()
                               if not isinstance(vv, np.ndarray)}
        else:
            save_results[k] = v

    with open(MODELS / 'full_experiment_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nAll results saved to {MODELS / 'full_experiment_results.json'}")


if __name__ == '__main__':
    main()
