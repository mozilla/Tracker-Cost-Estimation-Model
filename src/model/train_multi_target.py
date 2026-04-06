"""
Multi-target XGBoost training with TF-IDF URL embeddings.

Trains XGBoost Tweedie models for each target (transfer_bytes, load_ms,
download_ms, ttfb_ms) using hand-crafted features + TF-IDF URL embeddings.

Usage:
  python src/model/train_multi_target.py --data data/raw/per_request_1pct.csv
"""

import argparse
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

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output"
MODELS = ROOT / "models" / "per_request"

TARGETS = {
    'transfer_bytes': {
        'objective': 'reg:tweedie',
        'tweedie_variance_power': 1.5,
        'clip_min': 0,
        'description': 'On-wire response size (bytes)',
    },
    'load_ms': {
        'objective': 'reg:tweedie',
        'tweedie_variance_power': 1.3,  # less zero-inflated, lower power
        'clip_min': 0,
        'description': 'Total request duration (ms)',
    },
    'download_ms': {
        'objective': 'reg:tweedie',
        'tweedie_variance_power': 1.5,  # zero-inflated like transfer_bytes
        'clip_min': 0,
        'description': 'Download duration (ms)',
    },
    'ttfb_ms': {
        'objective': 'reg:tweedie',
        'tweedie_variance_power': 1.3,
        'clip_min': 0,
        'description': 'Time to first byte (ms)',
    },
}


def engineer_features(df, train_df, url_embeddings=None):
    """Build feature matrix from raw data + optional URL embeddings."""
    features = pd.DataFrame(index=df.index)

    # Target-encode domain (from train split only) — vectorized
    for target_col in ['transfer_bytes']:
        domain_medians = train_df.groupby('tracker_domain')[target_col].median()
        domain_type_medians = train_df.groupby(
            ['tracker_domain', 'resource_type'])[target_col].median()
        global_median = train_df[target_col].median()

        features['domain_median_bytes'] = (
            df['tracker_domain'].map(domain_medians).fillna(global_median))

        # Vectorized domain+type target encoding via merge
        dt_df = domain_type_medians.rename('_dt_med').reset_index()
        merged = df[['tracker_domain', 'resource_type']].merge(
            dt_df, on=['tracker_domain', 'resource_type'], how='left')
        # Fallback to domain median, then global
        merged['_dt_med'] = merged['_dt_med'].fillna(
            merged['tracker_domain'].map(domain_medians))
        features['domain_type_median'] = merged['_dt_med'].fillna(global_median).values

    # URL structure
    features['path_depth'] = df['path_depth'].fillna(0)
    features['url_length'] = df['url_length'].fillna(0)
    features['num_query_params'] = df['num_query_params'].fillna(0)
    features['has_query_params'] = df['has_query_params'].astype(int)

    # Resource type one-hot
    for rt in ['script', 'image', 'other', 'html', 'text', 'css']:
        features[f'rt_{rt}'] = (df['resource_type'] == rt).astype(int)

    # File extension one-hot
    for ext in ['js', 'gif', 'png', 'jpg', 'html', 'php', 'json', 'css']:
        features[f'ext_{ext}'] = (df['file_extension'] == ext).astype(int)

    # Initiator type
    for it in ['script', 'parser', 'other']:
        features[f'init_{it}'] = (df['initiator_type'] == it).astype(int)

    # HTTP method
    features['is_post'] = (df['http_method'] == 'POST').astype(int)

    # URL content regex features (hand-crafted baseline)
    url = df['url_path'].fillna('')
    features['path_has_js'] = url.str.contains(
        r'\.js|/js/|script|sdk|lib|tag|gtm|gtag', regex=True).astype(int)
    features['path_has_collect'] = url.str.contains(
        r'collect|beacon|ping|pixel|track', regex=True).astype(int)
    features['path_has_image'] = url.str.contains(
        r'\.gif|\.png|\.jpg|pixel|1x1', regex=True).astype(int)
    features['path_has_sync'] = url.str.contains(
        r'sync|match|cookie|usersync', regex=True).astype(int)
    features['path_has_ad'] = url.str.contains(
        r'/ad/|/ads/|adserver|pagead|prebid', regex=True).astype(int)
    features['path_has_api'] = url.str.contains(
        r'/api/|/v[0-9]/|/collect|/event', regex=True).astype(int)

    # URL embeddings (TF-IDF + SVD)
    if url_embeddings is not None:
        embed_df = pd.DataFrame(
            url_embeddings,
            columns=[f'url_emb_{i}' for i in range(url_embeddings.shape[1])],
            index=df.index,
        )
        features = pd.concat([features, embed_df], axis=1)

    return features


def lut_baseline(train_df, test_df, target_col):
    """Domain + resource type lookup table baseline."""
    # Compute medians from train
    domain_type_med = train_df.groupby(
        ['tracker_domain', 'resource_type'])[target_col].median()
    domain_med = train_df.groupby('tracker_domain')[target_col].median()
    global_med = train_df[target_col].median()

    # Predict with fallback chain
    preds = []
    for _, row in test_df[['tracker_domain', 'resource_type']].iterrows():
        key = (row['tracker_domain'], row['resource_type'])
        if key in domain_type_med.index:
            preds.append(domain_type_med[key])
        elif row['tracker_domain'] in domain_med.index:
            preds.append(domain_med[row['tracker_domain']])
        else:
            preds.append(global_med)
    return np.array(preds)


def lut_baseline_fast(train_df, test_df, target_col):
    """Vectorized LUT baseline — much faster for large datasets."""
    domain_type_med = train_df.groupby(
        ['tracker_domain', 'resource_type'])[target_col].median()
    domain_med = train_df.groupby('tracker_domain')[target_col].median()
    global_med = train_df[target_col].median()

    idx = pd.MultiIndex.from_frame(test_df[['tracker_domain', 'resource_type']])
    preds = pd.Series([domain_type_med.get(k, np.nan) for k in idx], index=test_df.index)

    # Fallback to domain median
    mask = preds.isna()
    preds[mask] = test_df.loc[mask, 'tracker_domain'].map(domain_med)

    # Fallback to global median
    preds = preds.fillna(global_med)

    return preds.values


def train_target(target_col, target_config, train_df, val_df, test_df,
                 X_train, X_val, X_test):
    """Train and evaluate XGBoost for one target."""
    print(f"\n{'='*70}")
    print(f"TARGET: {target_col} — {target_config['description']}")
    print(f"{'='*70}")

    # Clean target
    y_train = train_df[target_col].copy()
    y_val = val_df[target_col].copy()
    y_test = test_df[target_col].copy()

    # Drop rows with null targets
    train_mask = y_train.notna()
    val_mask = y_val.notna()
    test_mask = y_test.notna()

    y_train = y_train[train_mask].clip(lower=target_config['clip_min'])
    y_val = y_val[val_mask].clip(lower=target_config['clip_min'])
    y_test = y_test[test_mask].clip(lower=target_config['clip_min'])

    X_tr = X_train[train_mask]
    X_v = X_val[val_mask]
    X_te = X_test[test_mask]

    print(f"Train: {len(y_train):,}, Val: {len(y_val):,}, Test: {len(y_test):,}")
    print(f"Target stats — mean: {y_train.mean():.1f}, median: {y_train.median():.1f}, "
          f"zeros: {(y_train == 0).mean()*100:.1f}%")

    # LUT baseline
    t0 = time.time()
    lut_preds = lut_baseline_fast(
        train_df[train_mask.reindex(train_df.index, fill_value=True)],
        test_df[test_mask],
        target_col
    )
    lut_mae = mean_absolute_error(y_test, lut_preds)
    lut_med_ae = median_absolute_error(y_test, lut_preds)
    lut_rho_result = spearmanr(y_test, lut_preds)
    lut_rho = lut_rho_result.statistic if np.isfinite(lut_rho_result.statistic) else 0.0
    print(f"\nLUT baseline: MAE={lut_mae:,.0f}, MedAE={lut_med_ae:,.0f}, "
          f"rho={lut_rho:.4f} ({time.time()-t0:.1f}s)")

    # XGBoost with Tweedie loss
    # For Tweedie, targets must be >= 0. Add +1 offset for zero-inflated targets.
    offset = 1 if (y_train == 0).any() else 0

    params = dict(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=10,
        tree_method='hist',
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )

    if target_config['objective'] == 'reg:tweedie':
        params['objective'] = 'reg:tweedie'
        params['tweedie_variance_power'] = target_config['tweedie_variance_power']

    print(f"\nTraining XGBoost ({params['objective']}, "
          f"power={target_config.get('tweedie_variance_power', 'N/A')})...")
    t0 = time.time()

    model = xgb.XGBRegressor(early_stopping_rounds=20, **params)
    model.fit(
        X_tr, y_train + offset,
        eval_set=[(X_v, y_val + offset)],
        verbose=False,
    )
    train_time = time.time() - t0
    best_iter = getattr(model, 'best_iteration', params['n_estimators'])
    print(f"Trained in {train_time:.1f}s, best iteration: {best_iter}")

    # Predict
    preds = np.clip(model.predict(X_te) - offset, target_config['clip_min'], None)

    # Metrics
    mae = mean_absolute_error(y_test, preds)
    med_ae = median_absolute_error(y_test, preds)
    rho, _ = spearmanr(y_test, preds)
    improvement = (1 - mae / lut_mae) * 100

    print(f"\nXGBoost:  MAE={mae:,.0f}, MedAE={med_ae:,.0f}, rho={rho:.4f}")
    print(f"LUT:      MAE={lut_mae:,.0f}, MedAE={lut_med_ae:,.0f}, rho={lut_rho:.4f}")
    print(f"Improvement: {improvement:+.1f}% MAE")

    # Per resource type breakdown
    print(f"\nPer resource type:")
    print(f"  {'Type':<10s} {'Count':>8s} {'MAE':>10s} {'LUT MAE':>10s} {'Improv':>8s} {'rho':>8s}")
    print(f"  {'-'*54}")
    rt_results = {}
    for rt in ['script', 'image', 'other', 'html', 'text']:
        rt_mask = test_df.loc[test_mask.index[test_mask], 'resource_type'] == rt
        if rt_mask.sum() < 100:
            continue
        rt_mae = mean_absolute_error(y_test[rt_mask], preds[rt_mask.values])
        rt_lut = mean_absolute_error(y_test[rt_mask], lut_preds[rt_mask.values])
        rt_rho_result = spearmanr(y_test[rt_mask], preds[rt_mask.values])
        rt_rho = rt_rho_result.statistic if np.isfinite(rt_rho_result.statistic) else 0.0
        rt_improv = (1 - rt_mae / rt_lut) * 100 if rt_lut > 0 else 0
        print(f"  {rt:<10s} {rt_mask.sum():>8,d} {rt_mae:>10,.0f} {rt_lut:>10,.0f} "
              f"{rt_improv:>+7.1f}% {rt_rho:>8.4f}")
        rt_results[rt] = {'mae': float(rt_mae), 'lut_mae': float(rt_lut),
                          'rho': float(rt_rho), 'count': int(rt_mask.sum())}

    # Save model
    model_path = MODELS / f'xgb_{target_col}.json'
    model.save_model(str(model_path))
    print(f"\nModel saved to {model_path}")

    # Feature importance (top 15)
    importance = model.feature_importances_
    feature_names = X_tr.columns.tolist()
    top_idx = np.argsort(importance)[-15:][::-1]
    print(f"\nTop 15 features:")
    for i in top_idx:
        print(f"  {feature_names[i]:<25s} {importance[i]:.4f}")

    return {
        'target': target_col,
        'description': target_config['description'],
        'n_train': len(y_train),
        'n_test': len(y_test),
        'lut_mae': float(lut_mae),
        'lut_med_ae': float(lut_med_ae),
        'lut_rho': float(lut_rho),
        'xgb_mae': float(mae),
        'xgb_med_ae': float(med_ae),
        'xgb_rho': float(rho),
        'improvement_pct': float(improvement),
        'train_time_s': float(train_time),
        'best_iteration': int(best_iter),
        'per_resource_type': rt_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
                        default=str(ROOT / 'data' / 'raw' / 'per_request_1pct.csv'))
    parser.add_argument('--targets', type=str, nargs='+',
                        default=list(TARGETS.keys()),
                        help='Targets to train (default: all)')
    parser.add_argument('--no-embeddings', action='store_true',
                        help='Skip URL embeddings (hand-crafted features only)')
    args = parser.parse_args()

    OUTPUT.mkdir(exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {args.data}")
    t0 = time.time()
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    # Train/val/test split — row-level random split
    # Row-level is correct because Firefox will have HTTP Archive stats for all
    # Disconnect domains at deployment. Domain-level splits would destroy target
    # encoding features for unseen test domains, which is not the deployment scenario.
    np.random.seed(42)
    n = len(df)
    idx = np.random.permutation(n)
    train_idx = idx[:int(0.7 * n)]
    val_idx = idx[int(0.7 * n):int(0.85 * n)]
    test_idx = idx[int(0.85 * n):]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    print(f"Split: train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")
    print(f"Domains in test: {test_df['tracker_domain'].nunique():,} "
          f"(of {df['tracker_domain'].nunique():,} total)")

    # URL embeddings
    url_embed_train = None
    url_embed_val = None
    url_embed_test = None

    if not args.no_embeddings:
        print("\nFitting URL embeddings (TF-IDF + SVD)...")
        embedder = URLEmbedder(n_components=50)
        t0 = time.time()
        embedder.fit(train_df['url_path'].fillna(''))
        print(f"Fitted in {time.time()-t0:.1f}s, "
              f"variance explained: {embedder.total_variance_explained_*100:.1f}%")

        url_embed_train = embedder.transform(train_df['url_path'].fillna(''))
        url_embed_val = embedder.transform(val_df['url_path'].fillna(''))
        url_embed_test = embedder.transform(test_df['url_path'].fillna(''))
        embedder.save()

    # Feature engineering
    print("\nEngineering features...")
    X_train = engineer_features(train_df, train_df, url_embed_train)
    X_val = engineer_features(val_df, train_df, url_embed_val)
    X_test = engineer_features(test_df, train_df, url_embed_test)
    print(f"Feature matrix: {X_train.shape[1]} features")

    # Train each target
    all_results = {}
    for target_col in args.targets:
        if target_col not in TARGETS:
            print(f"Unknown target: {target_col}, skipping")
            continue
        result = train_target(
            target_col, TARGETS[target_col],
            train_df, val_df, test_df,
            X_train, X_val, X_test,
        )
        all_results[target_col] = result

    # Summary table
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Target':<18s} {'LUT MAE':>10s} {'XGB MAE':>10s} {'Improv':>8s} {'XGB rho':>8s}")
    print(f"{'-'*54}")
    for target_col, r in all_results.items():
        print(f"{target_col:<18s} {r['lut_mae']:>10,.0f} {r['xgb_mae']:>10,.0f} "
              f"{r['improvement_pct']:>+7.1f}% {r['xgb_rho']:>8.4f}")

    # Save results
    results_path = MODELS / 'multi_target_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
