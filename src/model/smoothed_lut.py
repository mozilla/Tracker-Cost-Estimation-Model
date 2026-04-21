"""
Smoothed LUT baseline experiment.

Implements a Laplace-smoothed (add-k) path-level lookup table and compares
to the unsmoothed version. This is the "real" comparator for the claim that
the model regularizes better than memorization.

Smoothing: for a path with n observations and sum S, predict:
    (S + k * global_mean) / (n + k)
where k is a smoothing hyperparameter. This interpolates between the raw path
median (k=0) and the global mean (k→∞).

We use mean rather than median for smoothed estimates (easier to smooth),
and test k in [0.5, 1, 2, 5, 10, 20].

Usage:
    python src/model/smoothed_lut.py > logs/smoothed_lut.log 2>&1
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw"
MODELS = ROOT / "models" / "per_request"

def run():
    print("Loading data...")
    df = pd.read_csv(DATA / "per_request_1pct.csv", low_memory=False)
    df = df[df["transfer_bytes"].notna() & (df["transfer_bytes"] >= 0)].copy()
    print(f"Loaded {len(df):,} requests, {df['tracker_domain'].nunique()} domains")

    # Row-level split (same seed as main paper)
    from sklearn.model_selection import train_test_split
    train_val, test = train_test_split(df, test_size=0.15, random_state=42)
    train, val = train_test_split(train_val, test_size=0.15/0.85, random_state=42)
    print(f"Split: train={len(train):,}, val={len(val):,}, test={len(test):,}")

    y_test = test["transfer_bytes"].values
    global_mean = train["transfer_bytes"].mean()
    global_median = train["transfer_bytes"].median()
    print(f"Global mean: {global_mean:.1f}, median: {global_median:.1f}")

    # --- Unsmoothed path LUT (median-based, same as paper) ---
    # path key: (tracker_domain, url_path)
    path_stats = train.groupby(["tracker_domain", "url_path"])["transfer_bytes"].agg(["median", "mean", "count"]).reset_index()
    path_stats.columns = ["tracker_domain", "url_path", "path_median", "path_mean", "path_count"]

    domain_type_stats = train.groupby(["tracker_domain", "resource_type"])["transfer_bytes"].median().reset_index()
    domain_type_stats.columns = ["tracker_domain", "resource_type", "domain_type_median"]

    domain_stats = train.groupby("tracker_domain")["transfer_bytes"].median().reset_index()
    domain_stats.columns = ["tracker_domain", "domain_median"]

    # Join fallbacks
    test2 = test.merge(path_stats, on=["tracker_domain", "url_path"], how="left")
    test2 = test2.merge(domain_type_stats, on=["tracker_domain", "resource_type"], how="left")
    test2 = test2.merge(domain_stats, on="tracker_domain", how="left")

    # Unsmoothed prediction (median with fallbacks)
    def predict_lut(row):
        if pd.notna(row.get("path_median")):
            return row["path_median"]
        elif pd.notna(row.get("domain_type_median")):
            return row["domain_type_median"]
        elif pd.notna(row.get("domain_median")):
            return row["domain_median"]
        return global_median

    print("Computing unsmoothed path LUT predictions...")
    pred_unsmoothed = test2.apply(predict_lut, axis=1).values
    mae_unsmoothed = mean_absolute_error(y_test, pred_unsmoothed)
    rho_unsmoothed = spearmanr(y_test, pred_unsmoothed).statistic

    n_seen = test2["path_median"].notna().sum()
    coverage = n_seen / len(test2)
    print(f"Unsmoothed path LUT: MAE={mae_unsmoothed:.1f}, rho={rho_unsmoothed:.3f}, coverage={coverage:.3f}")

    # Also compute MAE on seen vs unseen paths
    seen_mask = test2["path_median"].notna()
    mae_seen = mean_absolute_error(y_test[seen_mask], pred_unsmoothed[seen_mask])
    mae_unseen = mean_absolute_error(y_test[~seen_mask], pred_unsmoothed[~seen_mask])
    print(f"  Seen paths ({seen_mask.sum():,}): MAE={mae_seen:.1f}")
    print(f"  Unseen paths ({(~seen_mask).sum():,}): MAE={mae_unseen:.1f}")

    # --- Smoothed path LUT (mean-based with Laplace smoothing) ---
    # For seen paths: predict = (count * path_mean + k * global_mean) / (count + k)
    # For unseen paths: predict = domain_type fallback (unchanged)

    k_values = [0.5, 1, 2, 5, 10, 20, 50, 100]
    results = {
        "unsmoothed": {
            "mae": float(mae_unsmoothed),
            "rho": float(rho_unsmoothed),
            "coverage": float(coverage),
            "mae_seen": float(mae_seen),
            "mae_unseen": float(mae_unseen),
            "n_seen": int(n_seen),
            "n_unseen": int((~seen_mask).sum()),
        }
    }

    print("\nSmoothed LUT results:")
    print(f"{'k':>8} {'MAE':>10} {'MAE_seen':>12} {'MAE_unseen':>12} {'rho':>8}")
    print("-" * 54)

    for k in k_values:
        # Compute smoothed prediction for paths with data
        test2["smoothed_pred"] = np.where(
            test2["path_median"].notna(),
            (test2["path_count"] * test2["path_mean"] + k * global_mean) / (test2["path_count"] + k),
            np.where(
                test2["domain_type_median"].notna(),
                test2["domain_type_median"],
                np.where(
                    test2["domain_median"].notna(),
                    test2["domain_median"],
                    global_median
                )
            )
        )

        pred_smoothed = test2["smoothed_pred"].values
        mae_smoothed = mean_absolute_error(y_test, pred_smoothed)
        rho_smoothed = spearmanr(y_test, pred_smoothed).statistic

        mae_smoothed_seen = mean_absolute_error(y_test[seen_mask], pred_smoothed[seen_mask])
        mae_smoothed_unseen = mean_absolute_error(y_test[~seen_mask], pred_smoothed[~seen_mask])

        print(f"{k:>8.1f} {mae_smoothed:>10.1f} {mae_smoothed_seen:>12.1f} {mae_smoothed_unseen:>12.1f} {rho_smoothed:>8.3f}")

        results[f"k={k}"] = {
            "k": k,
            "mae": float(mae_smoothed),
            "rho": float(rho_smoothed),
            "mae_seen": float(mae_smoothed_seen),
            "mae_unseen": float(mae_smoothed_unseen),
        }

    # Best smoothed LUT
    best_k = min(k_values, key=lambda k: results[f"k={k}"]["mae"])
    best_mae = results[f"k={best_k}"]["mae"]
    print(f"\nBest smoothed LUT: k={best_k}, MAE={best_mae:.1f}")
    print(f"vs unsmoothed: {(mae_unsmoothed - best_mae)/mae_unsmoothed*100:+.1f}%")

    # Note: model MAE from paper is 3,466
    model_mae = 3465.65
    print(f"Model MAE (from paper): {model_mae:.1f}")
    print(f"Best smoothed LUT vs model: {(best_mae - model_mae)/model_mae*100:+.1f}%")

    results["model_mae"] = model_mae
    results["best_k"] = best_k
    results["best_smoothed_mae"] = best_mae

    out_path = MODELS / "smoothed_lut_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    run()
