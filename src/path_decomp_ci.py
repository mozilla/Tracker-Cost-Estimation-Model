"""
Bootstrap confidence intervals for path decomposition subsets.

Computes 95% bootstrap CIs (1,000 resamples) for Model MAE and Path LUT MAE on:
  - Seen-path subset (path LUT has exact training match)
  - Unseen-path subset (path LUT falls back to domain+type)

Usage:
  python3 src/path_decomp_ci.py
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
import json, sys

sys.path.insert(0, str(Path(__file__).parent / 'model'))
from url_embeddings import URLEmbedder
from train_multi_target import engineer_features

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models" / "per_request"

np.random.seed(42)
N_BOOT = 1000

print("Loading data...", flush=True)
df = pd.read_csv(ROOT / 'data' / 'raw' / 'per_request_1pct.csv', low_memory=False)
n = len(df)
idx = np.random.permutation(n)
train_df = df.iloc[idx[:int(0.7 * n)]].reset_index(drop=True)
test_df  = df.iloc[idx[int(0.85 * n):]].reset_index(drop=True)

print("Building features...", flush=True)
embedder = URLEmbedder(n_components=50)
embedder.fit(train_df['url_path'].fillna(''))
embed_test = embedder.transform(test_df['url_path'].fillna(''))
X_test = engineer_features(test_df, train_df, embed_test)

y_test = test_df['transfer_bytes'].clip(lower=0).values

# Build path LUT from training set
g       = train_df['transfer_bytes'].median()
dt_med  = train_df.groupby(['tracker_domain', 'resource_type'])['transfer_bytes'].median()
d_med   = train_df.groupby('tracker_domain')['transfer_bytes'].median()
path_med = train_df.groupby(['tracker_domain', 'url_path'])['transfer_bytes'].median()

# For each test row: is there an exact (domain, path) match in training?
path_df = path_med.rename('_pred').reset_index()
merged  = test_df[['tracker_domain', 'url_path', 'resource_type']].merge(
    path_df, on=['tracker_domain', 'url_path'], how='left')

path_lut_preds = merged['_pred'].values.copy()
seen_mask = ~np.isnan(path_lut_preds)

# Fill unseen rows with D+T LUT fallback
dt_df = dt_med.rename('_dt').reset_index()
dt_preds = test_df[['tracker_domain', 'resource_type']].merge(
    dt_df, on=['tracker_domain', 'resource_type'], how='left')['_dt']
dt_preds = dt_preds.fillna(test_df['tracker_domain'].map(d_med))
dt_preds = dt_preds.fillna(g).values

path_lut_preds[~seen_mask] = dt_preds[~seen_mask]

# Model predictions
print("Loading model and predicting...", flush=True)
model = xgb.XGBRegressor()
model.load_model(str(MODELS / 'xgb_transfer_bytes.json'))
model_preds = np.clip(model.predict(X_test) - 1, 0, None)

print(f"Seen paths: {seen_mask.sum():,}  ({seen_mask.mean()*100:.1f}%)")
print(f"Unseen paths: {(~seen_mask).sum():,}  ({(~seen_mask).mean()*100:.1f}%)")

def bootstrap_mae_ci(y_true, y_pred, n_boot=N_BOOT, alpha=0.05):
    n = len(y_true)
    mae_boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.random.randint(0, n, n)
        mae_boot[i] = np.mean(np.abs(y_true[idx] - y_pred[idx]))
    lo = np.percentile(mae_boot, 100 * alpha / 2)
    hi = np.percentile(mae_boot, 100 * (1 - alpha / 2))
    mae_pt = np.mean(np.abs(y_true - y_pred))
    return mae_pt, lo, hi

results = {}
for name, mask in [('seen', seen_mask), ('unseen', ~seen_mask)]:
    yt = y_test[mask]
    ym = model_preds[mask]
    yp = path_lut_preds[mask]

    print(f"\nBootstrapping {name} ({mask.sum():,} rows)...", flush=True)
    m_pt, m_lo, m_hi = bootstrap_mae_ci(yt, ym)
    p_pt, p_lo, p_hi = bootstrap_mae_ci(yt, yp)

    results[name] = {
        'n': int(mask.sum()),
        'model':    {'mae': m_pt, 'ci_lo': m_lo, 'ci_hi': m_hi},
        'path_lut': {'mae': p_pt, 'ci_lo': p_lo, 'ci_hi': p_hi},
    }
    print(f"  Model:    MAE={m_pt:.0f}  CI=[{m_lo:.0f}, {m_hi:.0f}]")
    print(f"  Path LUT: MAE={p_pt:.0f}  CI=[{p_lo:.0f}, {p_hi:.0f}]")
    overlap = not (m_hi < p_lo or p_hi < m_lo)
    print(f"  CIs overlap: {overlap}")

out = ROOT / "logs" / "path_decomp_ci_cycle14.json"
out.parent.mkdir(exist_ok=True)
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out}", flush=True)
