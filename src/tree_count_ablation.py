"""
Tree-count ablation: evaluate MAE at 50/100/200/300/500 trees using iteration_range.
Uses the already-trained 500-tree model — no retraining needed.

Output: logs/tree_count_ablation.json
"""

import json
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "per_request" / "xgb_transfer_bytes.json"
DATA_PATH = ROOT / "data" / "raw" / "per_request_1pct.csv"
EMBED_PATH = ROOT / "models" / "per_request" / "url_embedder.joblib"
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)
OUT = LOGS / "tree_count_ablation.json"

sys.path.insert(0, str(Path(__file__).parent / "model"))
from train_multi_target import engineer_features

print("Loading data...")
df = pd.read_csv(DATA_PATH, low_memory=False)
n = len(df)

# Replicate the 70/15/15 split from training
np.random.seed(42)
indices = np.random.permutation(n)
train_end = int(0.70 * n)
val_end = int(0.85 * n)
train_idx = indices[:train_end]
test_idx = indices[val_end:]

df_train = df.iloc[train_idx].reset_index(drop=True)
df_test = df.iloc[test_idx].reset_index(drop=True)
print(f"Train: {len(df_train)}, Test: {len(df_test)}")

# Compute URL embeddings for the test split using the saved embedder
print("Computing URL embeddings for test split...")
import joblib
embedder = joblib.load(str(EMBED_PATH))
test_embeddings = embedder.transform(df_test["url_path"].fillna(""))
print(f"Embeddings shape: {test_embeddings.shape}")

# Engineer features
print("Engineering test features...")
X_test = engineer_features(df_test, df_train, url_embeddings=test_embeddings)
y_test = df_test['transfer_bytes'].fillna(0).values
print(f"Feature matrix: {X_test.shape}, target: {y_test.shape}")

# Load model
print(f"Loading model from {MODEL_PATH}...")
booster = xgb.Booster()
booster.load_model(str(MODEL_PATH))
actual_trees = booster.num_boosted_rounds()
print(f"Model has {actual_trees} boosting rounds")

dtest = xgb.DMatrix(X_test)
results = []

for n_trees in [50, 100, 200, 300, 500]:
    nt = min(n_trees, actual_trees)
    preds_raw = booster.predict(dtest, iteration_range=(0, nt))
    # The model was trained with +1 offset on targets, reverse it
    preds = np.clip(preds_raw - 1, 0, None)
    mae = mean_absolute_error(y_test, preds)
    # Rough model size estimate: proportional to tree count (500 trees ~ 500KB)
    est_size_kb = max(10, int(nt / max(actual_trees, 1) * 500))
    results.append({
        "n_trees": nt,
        "mae": round(float(mae), 1),
        "est_size_kb": est_size_kb,
    })
    print(f"  n_trees={nt:3d}: MAE={mae:.1f}, est_size≈{est_size_kb}KB")

# Compute degradation vs full model
best_mae = results[-1]["mae"]
for r in results:
    r["pct_worse_than_500"] = round((r["mae"] - best_mae) / best_mae * 100, 1)

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {OUT}")
print(json.dumps(results, indent=2))
