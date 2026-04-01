"""
Advanced per-request prediction techniques.

Runs alongside the baseline experiment. Techniques:
1. Two-stage model (classify zero/nonzero, then regress on nonzero)
2. URL path tokenization + hashing features
3. Quantile regression for prediction intervals
4. Custom loss weighting (proportional to request size)

Usage:
  python src/model/train_advanced.py
"""

import numpy as np
import pandas as pd
import json
import warnings
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_extraction.text import HashingVectorizer
import xgboost as xgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw"
MODELS = ROOT / "models" / "per_request"
OUTPUT = ROOT / "output"

# Reuse feature definitions from main pipeline
from train_per_request import (
    load_data, row_level_split, engineer_features, evaluate,
    lut_baseline, RESOURCE_TYPES, URL_CONTENT_PATTERNS,
)


# ============================================================
# URL Path Tokenization
# ============================================================

def tokenize_url_path(path):
    """Convert URL path to space-separated tokens for hashing."""
    if pd.isna(path) or path == "":
        return ""
    # Split on /, ., -, _
    import re
    tokens = re.split(r'[/.\-_]', path)
    # Remove empty tokens and very long tokens (likely IDs/hashes)
    tokens = [t.lower() for t in tokens if t and len(t) < 20]
    return " ".join(tokens)


def add_url_hash_features(train_df, eval_df, n_features=64):
    """Add hashed URL path token features."""
    hasher = HashingVectorizer(
        n_features=n_features,
        token_pattern=r'[a-zA-Z]{2,}',
        alternate_sign=False,
    )

    train_paths = train_df["url_path"].fillna("").apply(tokenize_url_path)
    eval_paths = eval_df["url_path"].fillna("").apply(tokenize_url_path)

    # Fit on train (HashingVectorizer is stateless, but good practice)
    train_hashed = hasher.transform(train_paths).toarray()
    eval_hashed = hasher.transform(eval_paths).toarray()

    hash_cols = [f"url_hash_{i}" for i in range(n_features)]
    train_hash_df = pd.DataFrame(train_hashed, index=train_df.index, columns=hash_cols)
    eval_hash_df = pd.DataFrame(eval_hashed, index=eval_df.index, columns=hash_cols)

    return train_hash_df, eval_hash_df, hash_cols


# ============================================================
# Two-Stage Model
# ============================================================

def train_two_stage(X_train, y_train, X_val, y_val, n_trials=80):
    """
    Stage 1: Classify zero vs nonzero transfer_bytes
    Stage 2: Regress on nonzero requests only
    """
    print("\n  --- Stage 1: Zero/Nonzero Classifier ---")
    y_binary = (y_train > 0).astype(int)

    def cls_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 5.0),
        }
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr_idx, vl_idx in kf.split(X_train):
            model = xgb.XGBClassifier(**params, objective="binary:logistic",
                                       tree_method="hist", random_state=42, verbosity=0,
                                       eval_metric="logloss")
            model.fit(X_train.iloc[tr_idx], y_binary.iloc[tr_idx])
            from sklearn.metrics import f1_score
            preds = model.predict(X_train.iloc[vl_idx])
            scores.append(f1_score(y_binary.iloc[vl_idx], preds))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(cls_objective, n_trials=n_trials)
    print(f"  Classifier best F1: {study.best_value:.3f}")

    classifier = xgb.XGBClassifier(**study.best_params, objective="binary:logistic",
                                    tree_method="hist", random_state=42, verbosity=0,
                                    eval_metric="logloss")
    classifier.fit(X_train, y_binary)

    # Classifier accuracy on val
    val_binary_pred = classifier.predict(X_val)
    val_binary_true = (y_val > 0).astype(int)
    from sklearn.metrics import accuracy_score, f1_score
    print(f"  Val accuracy: {accuracy_score(val_binary_true, val_binary_pred):.3f}")
    print(f"  Val F1: {f1_score(val_binary_true, val_binary_pred):.3f}")

    print("\n  --- Stage 2: Regressor on Nonzero ---")
    nonzero_mask = y_train > 0
    X_train_nz = X_train[nonzero_mask]
    y_train_nz = y_train[nonzero_mask]

    def reg_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
        }
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr_idx, vl_idx in kf.split(X_train_nz):
            model = xgb.XGBRegressor(**params, objective="reg:squarederror",
                                      tree_method="hist", random_state=42, verbosity=0)
            model.fit(X_train_nz.iloc[tr_idx], y_train_nz.iloc[tr_idx])
            preds = np.clip(model.predict(X_train_nz.iloc[vl_idx]), 0, None)
            scores.append(mean_absolute_error(y_train_nz.iloc[vl_idx], preds))
        return np.mean(scores)

    study2 = optuna.create_study(direction="minimize")
    study2.optimize(reg_objective, n_trials=n_trials)
    print(f"  Regressor best CV MAE: {study2.best_value:,.0f}")

    regressor = xgb.XGBRegressor(**study2.best_params, objective="reg:squarederror",
                                  tree_method="hist", random_state=42, verbosity=0)
    regressor.fit(X_train_nz, y_train_nz)

    # Combined prediction
    def predict_two_stage(X):
        is_nonzero = classifier.predict(X)
        preds = np.zeros(len(X))
        nz_mask = is_nonzero == 1
        if nz_mask.sum() > 0:
            preds[nz_mask] = np.clip(regressor.predict(X[nz_mask]), 0, None)
        return preds

    return predict_two_stage, classifier, regressor


# ============================================================
# Quantile Regression
# ============================================================

def train_quantile_models(X_train, y_train, best_params, quantiles=[0.1, 0.5, 0.9]):
    """Train quantile regression models for prediction intervals."""
    models = {}
    for q in quantiles:
        print(f"  Training quantile={q}...")
        params = best_params.copy()
        model = xgb.XGBRegressor(
            **params,
            objective="reg:quantileerror",
            quantile_alpha=q,
            tree_method="hist",
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train)
        models[q] = model
    return models


def evaluate_intervals(y_true, pred_lo, pred_mid, pred_hi):
    coverage = np.mean((y_true >= pred_lo) & (y_true <= pred_hi))
    width = np.mean(pred_hi - pred_lo)
    median_width = np.median(pred_hi - pred_lo)
    return {
        "coverage": float(coverage),
        "mean_width": float(width),
        "median_width": float(median_width),
    }


# ============================================================
# XGBoost with URL hash features
# ============================================================

def train_xgb_with_url_hashing(X_train, y_train, X_train_hash, n_trials=100):
    """XGBoost with URL hash features appended."""
    X_combined = pd.concat([X_train.reset_index(drop=True), X_train_hash.reset_index(drop=True)], axis=1)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
        }
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr_idx, vl_idx in kf.split(X_combined):
            model = xgb.XGBRegressor(**params, objective="reg:squarederror",
                                      tree_method="hist", random_state=42, verbosity=0)
            model.fit(X_combined.iloc[tr_idx], y_train.iloc[tr_idx])
            preds = np.clip(model.predict(X_combined.iloc[vl_idx]), 0, None)
            scores.append(mean_absolute_error(y_train.iloc[vl_idx], preds))
        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print(f"  XGBoost+URLHash best CV MAE: {study.best_value:,.0f}")

    best = xgb.XGBRegressor(**study.best_params, objective="reg:squarederror",
                             tree_method="hist", random_state=42, verbosity=0)
    best.fit(X_combined, y_train)
    return best, study.best_params


# ============================================================
# Main
# ============================================================

def main():
    MODELS.mkdir(parents=True, exist_ok=True)

    df = load_data()
    train_df, val_df, test_df = row_level_split(df)

    # Engineer base features
    print("\nEngineering features...")
    X_train, feature_cols = engineer_features(train_df, train_df)
    X_val, _ = engineer_features(train_df, val_df)
    X_test, _ = engineer_features(train_df, test_df)

    y_train = train_df["transfer_bytes"].reset_index(drop=True)
    y_val = val_df["transfer_bytes"].reset_index(drop=True)
    y_test = test_df["transfer_bytes"].reset_index(drop=True)

    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    # URL hash features
    print("Computing URL hash features...")
    train_hash, val_hash, hash_cols = add_url_hash_features(train_df, val_df, n_features=64)
    _, test_hash, _ = add_url_hash_features(train_df, test_df, n_features=64)
    train_hash = train_hash.reset_index(drop=True)
    val_hash = val_hash.reset_index(drop=True)
    test_hash = test_hash.reset_index(drop=True)

    # LUT baseline for comparison
    lut_val_preds = lut_baseline(train_df, val_df)
    lut_test_preds = lut_baseline(train_df, test_df)

    results = {}

    print(f"\n{'='*60}")
    print("BASELINE")
    print(f"{'='*60}")
    results["LUT"] = {
        "val": evaluate(y_val.values, lut_val_preds, "LUT (validation)"),
        "test": evaluate(y_test.values, lut_test_preds, "LUT (test)"),
    }

    # ============================================================
    # 1. Two-Stage Model
    # ============================================================
    print(f"\n{'='*60}")
    print("TECHNIQUE 1: Two-Stage Model (classify + regress)")
    print(f"{'='*60}")
    predict_fn, cls, reg = train_two_stage(X_train, y_train, X_val, y_val, n_trials=40)
    two_stage_val = predict_fn(X_val)
    two_stage_test = predict_fn(X_test)
    results["TwoStage"] = {
        "val": evaluate(y_val.values, two_stage_val, "TwoStage (validation)"),
        "test": evaluate(y_test.values, two_stage_test, "TwoStage (test)"),
    }

    # ============================================================
    # 2. XGBoost + URL Hash Features
    # ============================================================
    print(f"\n{'='*60}")
    print("TECHNIQUE 2: XGBoost + URL Path Hashing (64 features)")
    print(f"{'='*60}")
    xgb_hash_model, xgb_hash_params = train_xgb_with_url_hashing(
        X_train, y_train, train_hash, n_trials=50
    )
    X_val_combined = pd.concat([X_val.reset_index(drop=True), val_hash.reset_index(drop=True)], axis=1)
    X_test_combined = pd.concat([X_test.reset_index(drop=True), test_hash.reset_index(drop=True)], axis=1)
    hash_val_preds = np.clip(xgb_hash_model.predict(X_val_combined), 0, None)
    hash_test_preds = np.clip(xgb_hash_model.predict(X_test_combined), 0, None)
    results["XGB+URLHash"] = {
        "val": evaluate(y_val.values, hash_val_preds, "XGB+URLHash (validation)"),
        "test": evaluate(y_test.values, hash_test_preds, "XGB+URLHash (test)"),
    }

    # ============================================================
    # 3. Quantile Regression (prediction intervals)
    # ============================================================
    print(f"\n{'='*60}")
    print("TECHNIQUE 3: Quantile Regression (prediction intervals)")
    print(f"{'='*60}")

    # Use best params from the URL hash model (or default reasonable params)
    q_params = {
        "n_estimators": 200,
        "max_depth": 7,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    q_models = train_quantile_models(X_train, y_train, q_params)

    q_val_lo = np.clip(q_models[0.1].predict(X_val), 0, None)
    q_val_mid = np.clip(q_models[0.5].predict(X_val), 0, None)
    q_val_hi = np.clip(q_models[0.9].predict(X_val), 0, None)

    q_test_lo = np.clip(q_models[0.1].predict(X_test), 0, None)
    q_test_mid = np.clip(q_models[0.5].predict(X_test), 0, None)
    q_test_hi = np.clip(q_models[0.9].predict(X_test), 0, None)

    val_intervals = evaluate_intervals(y_val.values, q_val_lo, q_val_mid, q_val_hi)
    test_intervals = evaluate_intervals(y_test.values, q_test_lo, q_test_mid, q_test_hi)

    print(f"\n  Validation intervals (p10-p90):")
    print(f"    Coverage: {val_intervals['coverage']:.3f} (target ~0.80)")
    print(f"    Mean width: {val_intervals['mean_width']:,.0f} bytes")
    print(f"    Median width: {val_intervals['median_width']:,.0f} bytes")

    print(f"\n  Test intervals (p10-p90):")
    print(f"    Coverage: {test_intervals['coverage']:.3f}")
    print(f"    Mean width: {test_intervals['mean_width']:,.0f} bytes")

    results["Quantile_p50"] = {
        "val": evaluate(y_val.values, q_val_mid, "Quantile p50 (validation)"),
        "test": evaluate(y_test.values, q_test_mid, "Quantile p50 (test)"),
        "val_intervals": val_intervals,
        "test_intervals": test_intervals,
    }

    # ============================================================
    # 4. XGBoost with Tweedie loss (size-proportional errors)
    # ============================================================
    print(f"\n{'='*60}")
    print("TECHNIQUE 4: XGBoost with Tweedie Loss")
    print(f"{'='*60}")

    # Tweedie requires strictly positive targets, add small offset
    y_train_tw = y_train + 1  # shift to avoid zeros

    def tweedie_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.1, 1.9),
        }
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr_idx, vl_idx in kf.split(X_train):
            model = xgb.XGBRegressor(**params, objective="reg:tweedie",
                                      tree_method="hist", random_state=42, verbosity=0)
            model.fit(X_train.iloc[tr_idx], y_train_tw.iloc[tr_idx])
            preds = model.predict(X_train.iloc[vl_idx])
            preds = np.nan_to_num(preds, nan=0.0)
            preds = np.clip(preds - 1, 0, None)  # undo offset
            scores.append(mean_absolute_error(y_train.iloc[vl_idx], preds))
        return np.mean(scores)

    study_tw = optuna.create_study(direction="minimize")
    study_tw.optimize(tweedie_objective, n_trials=40)
    print(f"  Tweedie best CV MAE: {study_tw.best_value:,.0f}")

    tw_model = xgb.XGBRegressor(**study_tw.best_params, objective="reg:tweedie",
                                 tree_method="hist", random_state=42, verbosity=0)
    tw_model.fit(X_train, y_train_tw)
    tw_val_preds = np.nan_to_num(tw_model.predict(X_val), nan=0.0)
    tw_val_preds = np.clip(tw_val_preds - 1, 0, None)
    tw_test_preds = np.nan_to_num(tw_model.predict(X_test), nan=0.0)
    tw_test_preds = np.clip(tw_test_preds - 1, 0, None)
    results["XGB_Tweedie"] = {
        "val": evaluate(y_val.values, tw_val_preds, "XGB Tweedie (validation)"),
        "test": evaluate(y_test.values, tw_test_preds, "XGB Tweedie (test)"),
    }

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print("ADVANCED TECHNIQUES SUMMARY (validation set)")
    print(f"{'='*60}")
    print(f"\n{'Model':20s}  {'MAE':>10s}  {'MedAE':>8s}  {'RMSE':>10s}  {'Spearman':>10s}")
    print("-" * 65)
    for name, r in results.items():
        v = r["val"]
        print(f"{name:20s}  {v['mae']:>10,.0f}  {v['median_ae']:>8,.0f}  {v['rmse']:>10,.0f}  {v['spearman_rho']:>10.4f}")

    best_name = min(results.keys(), key=lambda k: results[k]["val"]["mae"])
    print(f"\nBest model (val MAE): {best_name}")

    print(f"\n{'='*60}")
    print("TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"\n{'Model':20s}  {'MAE':>10s}  {'MedAE':>8s}  {'RMSE':>10s}  {'Spearman':>10s}")
    print("-" * 65)
    for name, r in results.items():
        t = r["test"]
        print(f"{name:20s}  {t['mae']:>10,.0f}  {t['median_ae']:>8,.0f}  {t['rmse']:>10,.0f}  {t['spearman_rho']:>10.4f}")

    lut_mae = results["LUT"]["test"]["mae"]
    best_mae = results[best_name]["test"]["mae"]
    print(f"\nBest improvement over LUT: {(1 - best_mae / lut_mae) * 100:+.1f}%")

    # Save
    with open(MODELS / "advanced_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {MODELS / 'advanced_results.json'}")


if __name__ == "__main__":
    main()
