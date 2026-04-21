"""
Compute within-10% aggregation rates for all N values.

For N in {50, 100, 200, 500}, runs 2,000-trial i.i.d. aggregation simulation
and computes median % error and fraction within 10% for model and D+T LUT.

Usage:
  python3 src/within_10pct_agg.py
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
N_TRIALS = 2000

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

# D+T LUT predictions
g      = train_df['transfer_bytes'].median()
dt_med = train_df.groupby(['tracker_domain', 'resource_type'])['transfer_bytes'].median()
d_med  = train_df.groupby('tracker_domain')['transfer_bytes'].median()
dt_df  = dt_med.rename('_pred').reset_index()
lut_preds = test_df[['tracker_domain', 'resource_type']].merge(
    dt_df, on=['tracker_domain', 'resource_type'], how='left')['_pred']
lut_preds = lut_preds.fillna(test_df['tracker_domain'].map(d_med))
lut_preds = lut_preds.fillna(g).values

# Model predictions
print("Loading model and predicting...", flush=True)
model = xgb.XGBRegressor()
model.load_model(str(MODELS / 'xgb_transfer_bytes.json'))
model_preds = np.clip(model.predict(X_test) - 1, 0, None)

n_test = len(y_test)

def agg_stats(y_true, y_pred, N, n_trials=N_TRIALS):
    pct_errs = []
    for _ in range(n_trials):
        idx = np.random.choice(n_test, size=N, replace=True)
        true_sum = y_true[idx].sum()
        if true_sum == 0:
            continue
        pred_sum = y_pred[idx].sum()
        pct_errs.append(abs(pred_sum - true_sum) / true_sum * 100)
    pct_errs = np.array(pct_errs)
    return {
        'median_pct_err': float(np.median(pct_errs)),
        'within_10pct': float(np.mean(pct_errs <= 10.0)),
    }

results = {}
print("\nAggregation stats (uniform i.i.d. sampling):", flush=True)
print(f"{'N':>6s}  {'Model med%':>12s}  {'Model w10%':>12s}  {'LUT med%':>10s}  {'LUT w10%':>10s}")
print("-" * 60)
for N in [50, 100, 200, 500]:
    m = agg_stats(y_test, model_preds, N)
    l = agg_stats(y_test, lut_preds,  N)
    results[N] = {'model': m, 'lut': l}
    print(f"{N:>6d}  {m['median_pct_err']:>12.1f}  {m['within_10pct']*100:>11.1f}%  "
          f"{l['median_pct_err']:>10.1f}  {l['within_10pct']*100:>9.1f}%", flush=True)

out = ROOT / "logs" / "within_10pct_agg_cycle15.json"
out.parent.mkdir(exist_ok=True)
with open(out, 'w') as f:
    json.dump({str(k): v for k, v in results.items()}, f, indent=2)
print(f"\nResults saved to {out}", flush=True)
