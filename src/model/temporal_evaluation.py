"""
Temporal holdout evaluation.

Train on June 2024, test on September 2024.
Evaluates whether the model generalizes across time.

Usage:
  python src/model/temporal_evaluation.py
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


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000):
    np.random.seed(42)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    return np.mean(scores), np.percentile(scores, 2.5), np.percentile(scores, 97.5)


def aggregation_sim(y_true, y_pred, Ns=[50, 100, 200, 500], n_trials=2000):
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
        }
    return results


def domain_type_lut_predict(train_df, test_df, target_col):
    dt_med = train_df.groupby(['tracker_domain', 'resource_type'])[target_col].median()
    d_med = train_df.groupby('tracker_domain')[target_col].median()
    g_med = train_df[target_col].median()

    # Vectorized merge
    dt_df = dt_med.rename('_pred').reset_index()
    preds = test_df[['tracker_domain', 'resource_type']].merge(
        dt_df, on=['tracker_domain', 'resource_type'], how='left')['_pred']
    preds = preds.fillna(test_df['tracker_domain'].map(d_med))
    preds = preds.fillna(g_med)
    return preds.values


def path_lut_predict(train_df, test_df, target_col):
    path_med = train_df.groupby(['tracker_domain', 'url_path'])[target_col].median()
    dt_med = train_df.groupby(['tracker_domain', 'resource_type'])[target_col].median()
    d_med = train_df.groupby('tracker_domain')[target_col].median()
    g_med = train_df[target_col].median()

    # Vectorized path merge
    path_df = path_med.rename('_pred').reset_index()
    preds = test_df[['tracker_domain', 'url_path']].merge(
        path_df, on=['tracker_domain', 'url_path'], how='left')['_pred']
    path_hit = preds.notna()

    # Fallback: domain+type
    dt_df = dt_med.rename('_dt').reset_index()
    dt_preds = test_df[['tracker_domain', 'resource_type']].merge(
        dt_df, on=['tracker_domain', 'resource_type'], how='left')['_dt']
    preds = preds.fillna(dt_preds)

    # Fallback: domain, then global
    preds = preds.fillna(test_df['tracker_domain'].map(d_med))
    preds = preds.fillna(g_med)

    return preds.values, path_hit.mean()


def main():
    np.random.seed(42)

    # Load June (training) data
    print("Loading June 2024 training data...")
    t0 = time.time()
    june_df = pd.read_csv(ROOT / 'data' / 'raw' / 'per_request_1pct.csv', low_memory=False)
    print(f"  {len(june_df):,} rows in {time.time()-t0:.1f}s")

    # Load September (holdout) data
    print("Loading September 2024 holdout data...")
    t0 = time.time()
    sep_df = pd.read_csv(ROOT / 'data' / 'raw' / 'per_request_1pct_sep2024.csv', low_memory=False)
    print(f"  {len(sep_df):,} rows in {time.time()-t0:.1f}s")

    # Split June into train/val (same as main experiments)
    n = len(june_df)
    idx = np.random.permutation(n)
    train_df = june_df.iloc[idx[:int(0.7 * n)]].reset_index(drop=True)
    val_df = june_df.iloc[idx[int(0.7 * n):int(0.85 * n)]].reset_index(drop=True)
    june_test_df = june_df.iloc[idx[int(0.85 * n):]].reset_index(drop=True)

    # September is the full temporal holdout
    sep_test_df = sep_df.reset_index(drop=True)

    print(f"\nJune train: {len(train_df):,}")
    print(f"June test:  {len(june_test_df):,}")
    print(f"Sep test:   {len(sep_test_df):,}")

    # ============================================================
    # DRIFT ANALYSIS
    # ============================================================
    print(f"\n{'='*70}")
    print("TEMPORAL DRIFT ANALYSIS")
    print(f"{'='*70}")

    june_domains = set(train_df['tracker_domain'].unique())
    sep_domains = set(sep_test_df['tracker_domain'].unique())
    shared_domains = june_domains & sep_domains
    new_domains = sep_domains - june_domains
    print(f"June train domains: {len(june_domains):,}")
    print(f"Sep test domains:   {len(sep_domains):,}")
    print(f"Shared:             {len(shared_domains):,} ({len(shared_domains)/len(sep_domains)*100:.1f}%)")
    print(f"New in Sep:         {len(new_domains):,} ({len(new_domains)/len(sep_domains)*100:.1f}%)")

    # Path overlap
    june_paths = set(zip(train_df['tracker_domain'], train_df['url_path']))
    sep_paths = set(zip(sep_test_df['tracker_domain'], sep_test_df['url_path']))
    shared_paths = june_paths & sep_paths
    print(f"\nJune unique paths:  {len(june_paths):,}")
    print(f"Sep unique paths:   {len(sep_paths):,}")
    print(f"Shared:             {len(shared_paths):,} ({len(shared_paths)/len(sep_paths)*100:.1f}%)")

    # Row-level path coverage (vectorized via merge)
    june_path_df = pd.DataFrame(list(june_paths), columns=['tracker_domain', 'url_path'])
    june_path_df['_hit'] = True
    hit_merge = sep_test_df[['tracker_domain', 'url_path']].merge(
        june_path_df, on=['tracker_domain', 'url_path'], how='left')
    row_path_hit = hit_merge['_hit'].fillna(False).values
    print(f"Sep rows with June-seen path: {row_path_hit.mean()*100:.1f}%")

    # Distribution comparison
    print(f"\nTransfer size comparison:")
    print(f"  June train — mean: {train_df['transfer_bytes'].mean():,.0f}, median: {train_df['transfer_bytes'].median():.0f}")
    print(f"  Sep test   — mean: {sep_test_df['transfer_bytes'].mean():,.0f}, median: {sep_test_df['transfer_bytes'].median():.0f}")

    # ============================================================
    # FIT EMBEDDINGS AND FEATURES (on June train only)
    # ============================================================
    print(f"\n{'='*70}")
    print("FITTING FEATURES ON JUNE TRAINING DATA")
    print(f"{'='*70}")

    embedder = URLEmbedder(n_components=50)
    print("Fitting TF-IDF on June train...")
    embedder.fit(train_df['url_path'].fillna(''))

    embed_june_test = embedder.transform(june_test_df['url_path'].fillna(''))
    embed_sep_test = embedder.transform(sep_test_df['url_path'].fillna(''))

    X_june_test = engineer_features(june_test_df, train_df, embed_june_test)
    X_sep_test = engineer_features(sep_test_df, train_df, embed_sep_test)

    y_june = june_test_df['transfer_bytes'].clip(lower=0).values
    y_sep = sep_test_df['transfer_bytes'].clip(lower=0).values

    # ============================================================
    # TRAIN MODEL ON JUNE (or load existing)
    # ============================================================
    model_path = MODELS / 'xgb_transfer_bytes_temporal.json'
    embed_train = embedder.transform(train_df['url_path'].fillna(''))
    embed_val = embedder.transform(val_df['url_path'].fillna(''))
    X_train = engineer_features(train_df, train_df, embed_train)
    X_val = engineer_features(val_df, train_df, embed_val)
    y_train = train_df['transfer_bytes'].clip(lower=0)
    y_val = val_df['transfer_bytes'].clip(lower=0)

    print("Training XGBoost Tweedie on June data...")
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
        objective='reg:tweedie', tweedie_variance_power=1.5,
        tree_method='hist', random_state=42, verbosity=0,
        n_jobs=-1, early_stopping_rounds=20,
    )
    model.fit(X_train, y_train + 1, eval_set=[(X_val, y_val + 1)], verbose=False)
    best_iter = getattr(model, 'best_iteration', 500)
    print(f"Trained, best iteration: {best_iter}")
    model.save_model(str(model_path))

    # ============================================================
    # EVALUATE: JUNE TEST vs SEPTEMBER TEST
    # ============================================================
    print(f"\n{'='*70}")
    print("RESULTS: JUNE (in-distribution) vs SEPTEMBER (temporal holdout)")
    print(f"{'='*70}")

    results = {}

    for name, test_df, X_test, y_test in [
        ('June (in-dist)', june_test_df, X_june_test, y_june),
        ('September (temporal)', sep_test_df, X_sep_test, y_sep),
    ]:
        print(f"\n--- {name} ---")

        # LUT baseline
        lut_preds = domain_type_lut_predict(train_df, test_df, 'transfer_bytes')
        lut_mae = mean_absolute_error(y_test, lut_preds)

        # Path LUT
        path_preds, path_cov = path_lut_predict(train_df, test_df, 'transfer_bytes')
        path_mae = mean_absolute_error(y_test, path_preds)

        # Model
        model_preds = np.clip(model.predict(X_test) - 1, 0, None)
        model_mae = mean_absolute_error(y_test, model_preds)
        model_rho = spearmanr(y_test, model_preds).statistic

        # Bootstrap CIs
        _, model_ci_lo, model_ci_hi = bootstrap_ci(y_test, model_preds, mean_absolute_error)

        # Improvement
        improv = (1 - model_mae / lut_mae) * 100

        print(f"  Domain+type LUT: MAE={lut_mae:,.0f}")
        print(f"  Path LUT:        MAE={path_mae:,.0f} (coverage={path_cov*100:.1f}%)")
        print(f"  XGB Tweedie:     MAE={model_mae:,.0f} [{model_ci_lo:,.0f}, {model_ci_hi:,.0f}]  rho={model_rho:.4f}")
        print(f"  vs LUT:          {improv:+.1f}%")

        # Aggregation
        agg = aggregation_sim(y_test, model_preds)
        agg_lut = aggregation_sim(y_test, lut_preds)
        print(f"  Aggregation (N=200): Model {agg[200]['median_pct_err']:.1f}%, LUT {agg_lut[200]['median_pct_err']:.1f}%")

        # Per resource type
        print(f"\n  Per resource type:")
        print(f"  {'Type':<10s} {'n':>8s} {'LUT MAE':>10s} {'Model MAE':>10s} {'Improv':>8s}")
        rt_results = {}
        for rt in ['script', 'image', 'other', 'html', 'text']:
            mask = test_df['resource_type'] == rt
            if mask.sum() < 100:
                continue
            rt_lut = mean_absolute_error(y_test[mask], lut_preds[mask])
            rt_mod = mean_absolute_error(y_test[mask], model_preds[mask])
            rt_imp = (1 - rt_mod / rt_lut) * 100 if rt_lut > 0 else 0
            print(f"  {rt:<10s} {mask.sum():>8,d} {rt_lut:>10,.0f} {rt_mod:>10,.0f} {rt_imp:>+7.1f}%")
            rt_results[rt] = {'n': int(mask.sum()), 'lut_mae': float(rt_lut),
                              'model_mae': float(rt_mod)}

        results[name] = {
            'n': len(y_test),
            'lut_mae': float(lut_mae),
            'path_mae': float(path_mae),
            'path_coverage': float(path_cov),
            'model_mae': float(model_mae),
            'model_ci': [float(model_ci_lo), float(model_ci_hi)],
            'model_rho': float(model_rho),
            'improvement_pct': float(improv),
            'agg_200_model': float(agg[200]['median_pct_err']),
            'agg_200_lut': float(agg_lut[200]['median_pct_err']),
            'per_resource_type': rt_results,
        }

    # ============================================================
    # DEGRADATION SUMMARY
    # ============================================================
    print(f"\n{'='*70}")
    print("DEGRADATION SUMMARY")
    print(f"{'='*70}")

    june_r = results['June (in-dist)']
    sep_r = results['September (temporal)']

    mae_degrad = (sep_r['model_mae'] - june_r['model_mae']) / june_r['model_mae'] * 100
    lut_degrad = (sep_r['lut_mae'] - june_r['lut_mae']) / june_r['lut_mae'] * 100
    print(f"Model MAE degradation:  {june_r['model_mae']:,.0f} → {sep_r['model_mae']:,.0f} ({mae_degrad:+.1f}%)")
    print(f"LUT MAE degradation:    {june_r['lut_mae']:,.0f} → {sep_r['lut_mae']:,.0f} ({lut_degrad:+.1f}%)")
    print(f"Model still beats LUT:  {sep_r['improvement_pct']:+.1f}%")
    print(f"Path coverage drop:     {june_r['path_coverage']*100:.1f}% → {sep_r['path_coverage']*100:.1f}%")
    print(f"Aggregation (N=200):    Model {june_r['agg_200_model']:.1f}% → {sep_r['agg_200_model']:.1f}%")

    # Drift stats
    results['drift'] = {
        'june_domains': len(june_domains),
        'sep_domains': len(sep_domains),
        'shared_domains': len(shared_domains),
        'new_domains': len(new_domains),
        'june_paths': len(june_paths),
        'sep_paths': len(sep_paths),
        'shared_paths': len(shared_paths),
        'row_path_coverage': float(row_path_hit.mean()),
        'model_mae_degradation_pct': float(mae_degrad),
        'lut_mae_degradation_pct': float(lut_degrad),
    }

    # Save
    MODELS.mkdir(parents=True, exist_ok=True)
    with open(MODELS / 'temporal_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {MODELS / 'temporal_evaluation_results.json'}")


if __name__ == '__main__':
    main()
