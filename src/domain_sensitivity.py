"""
Domain-count sensitivity for correlated browsing aggregation.

Tests model aggregation accuracy at N=200 for n_domains in {5, 10, 15, 20, 25}.
Answers the deferred question: how sensitive are the correlated browsing results
to the choice of 15 domains?

Usage:
  python src/domain_sensitivity.py
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent / 'model'))
from url_embeddings import URLEmbedder
from train_multi_target import engineer_features

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models" / "per_request"

np.random.seed(42)

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

y_test  = test_df['transfer_bytes'].clip(lower=0).values
domains = test_df['tracker_domain']

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
model = xgb.XGBRegressor()
model.load_model(str(MODELS / 'xgb_transfer_bytes.json'))
model_preds = np.clip(model.predict(X_test) - 1, 0, None)

unique_domains = domains.unique()
domain_indices = {d: np.where(domains.values == d)[0] for d in unique_domains}

def domain_corr_agg(y_true, y_pred, n_domains, N=200, n_trials=2000):
    errs = []
    for _ in range(n_trials):
        chosen = np.random.choice(unique_domains, size=n_domains, replace=True)
        pool = np.concatenate([domain_indices[d] for d in chosen])
        if len(pool) < N:
            idx = pool
        else:
            idx = np.random.choice(pool, size=N, replace=False)
        true_sum = y_true[idx].sum()
        if true_sum == 0:
            continue
        pred_sum = y_pred[idx].sum()
        errs.append(abs(pred_sum - true_sum) / true_sum * 100)
    return float(np.median(errs))

results = {}
print("\nDomain sensitivity at N=200, 2000 trials:", flush=True)
print(f"{'n_domains':>10s}  {'Model %':>10s}  {'D+T LUT %':>10s}  {'Gap pp':>10s}")
print("-" * 50)
for nd in [5, 10, 15, 20, 25]:
    m = domain_corr_agg(y_test, model_preds, nd)
    l = domain_corr_agg(y_test, lut_preds,  nd)
    results[nd] = {'model': m, 'lut': l, 'gap': l - m}
    print(f"{nd:>10d}  {m:>10.1f}  {l:>10.1f}  {l-m:>10.1f}", flush=True)

out = ROOT / "logs" / "domain_sensitivity_cycle13.json"
out.parent.mkdir(exist_ok=True)
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out}", flush=True)
