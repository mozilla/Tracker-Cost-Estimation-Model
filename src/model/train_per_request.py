"""
Full-scale per-request transfer size prediction.

Domain-level train/val/test split.
Target encoding computed from train only.
Models: LUT baseline, Ridge, Random Forest, XGBoost, LightGBM, CatBoost, MLP.
Hyperparameter tuning via Optuna on train set (5-fold CV).
Final evaluation on held-out test set.

Usage:
  python src/model/train_per_request.py
"""

import numpy as np
import pandas as pd
import json
import warnings
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw"
MODELS = ROOT / "models" / "per_request"
OUTPUT = ROOT / "output"

# ============================================================
# Feature definitions
# ============================================================

URL_CONTENT_PATTERNS = {
    "path_has_js": r"\.js|/js/|script|sdk|lib|tag|gtm|gtag",
    "path_has_collect": r"collect|beacon|ping|pixel|track",
    "path_has_image": r"\.gif|\.png|\.jpg|pixel|1x1",
    "path_has_sync": r"sync|match|cookie|usersync",
    "path_has_ad": r"/ad/|/ads/|adserver|pagead|prebid",
    "path_has_api": r"/api/|/v[0-9]/|/collect|/event",
}

RESOURCE_TYPES = ["script", "image", "other", "html", "text", "css", "video", "xml", "font"]
INITIATOR_TYPES = ["script", "parser", "other", "preflight"]
HTTP_METHODS = ["GET", "POST"]
HTTP_VERSIONS = ["HTTP/2", "h3", "http/1.1"]
EXT_GROUPS = {"js", "gif", "html", "php", "jpg", "json", "png", "css"}

PRIORITY_MAP = {"Lowest": 0, "Low": 1, "Medium": 2, "High": 3, "Highest": 4}


def load_data():
    df = pd.read_csv(DATA / "per_request_full.csv", low_memory=False)
    df = df[df["transfer_bytes"].notna() & (df["transfer_bytes"] >= 0)].copy()
    print(f"Loaded {len(df):,} requests, {df['tracker_domain'].nunique()} domains")
    return df


# ============================================================
# Domain-level split
# ============================================================

def row_level_split(df, train_frac=0.70, val_frac=0.15, seed=42):
    """Split by row (stratified by domain to ensure representation)."""
    from sklearn.model_selection import train_test_split

    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df, test_size=1.0 - train_frac - val_frac, random_state=seed,
    )
    # Second split: train vs val
    val_relative = val_frac / (train_frac + val_frac)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_relative, random_state=seed,
    )

    print(f"Split: train={len(train_df):,} ({train_df['tracker_domain'].nunique()} domains), "
          f"val={len(val_df):,} ({val_df['tracker_domain'].nunique()} domains), "
          f"test={len(test_df):,} ({test_df['tracker_domain'].nunique()} domains)")

    return train_df, val_df, test_df


# ============================================================
# Feature engineering
# ============================================================

def engineer_features(train_df, eval_df):
    """
    Engineer features for eval_df using statistics from train_df only.
    Returns (X_eval, feature_cols).
    """
    df = eval_df.copy()

    # --- Target encoding: domain median (from train) ---
    domain_medians = train_df.groupby("tracker_domain")["transfer_bytes"].median()
    global_median = train_df["transfer_bytes"].median()
    df["domain_median_bytes"] = df["tracker_domain"].map(domain_medians).fillna(global_median)

    # --- Target encoding: domain + type median (from train) ---
    dt_medians = train_df.groupby(["tracker_domain", "resource_type"])["transfer_bytes"].median().to_dict()
    df["domain_type_median"] = df.apply(
        lambda r: dt_medians.get(
            (r["tracker_domain"], r["resource_type"]),
            domain_medians.get(r["tracker_domain"], global_median),
        ),
        axis=1,
    )

    # --- Domain request volume (from train) ---
    domain_volume = train_df.groupby("tracker_domain").size()
    df["domain_volume"] = np.log1p(df["tracker_domain"].map(domain_volume).fillna(0))

    # --- One-hot: resource_type ---
    for rt in RESOURCE_TYPES:
        df[f"rt_{rt}"] = (df["resource_type"] == rt).astype(int)

    # --- One-hot: initiator_type ---
    for it in INITIATOR_TYPES:
        df[f"init_{it}"] = (df["initiator_type"] == it).astype(int)

    # --- Ordinal: chrome_priority ---
    df["priority_ord"] = df["chrome_priority"].map(PRIORITY_MAP).fillna(1)

    # --- One-hot: http_method ---
    for m in HTTP_METHODS:
        df[f"method_{m}"] = (df["http_method"] == m).astype(int)

    # --- One-hot: http_version ---
    for v in HTTP_VERSIONS:
        df[f"httpv_{v}"] = (df["http_version"] == v).astype(int)

    # --- File extension groups ---
    df["ext_clean"] = df["file_extension"].apply(lambda x: x if x in EXT_GROUPS else "other")
    df["ext_clean"] = df["ext_clean"].fillna("none")
    for e in list(EXT_GROUPS) + ["other", "none"]:
        df[f"ext_{e}"] = (df["ext_clean"] == e).astype(int)

    # --- URL content signals ---
    path = df["url_path"].fillna("")
    for name, pattern in URL_CONTENT_PATTERNS.items():
        df[name] = path.str.contains(pattern, case=False, regex=True).astype(int)

    # --- Numeric features ---
    df["path_token_count"] = df["url_path"].fillna("").str.count("/")

    # --- Feature columns ---
    feature_cols = (
        ["domain_median_bytes", "domain_type_median", "domain_volume"]
        + ["path_depth", "url_length", "num_query_params", "has_query_params", "path_token_count"]
        + [f"ext_{e}" for e in list(EXT_GROUPS) + ["other", "none"]]
        + list(URL_CONTENT_PATTERNS.keys())
        + [f"rt_{rt}" for rt in RESOURCE_TYPES]
        + [f"init_{it}" for it in INITIATOR_TYPES]
        + ["priority_ord"]
        + [f"method_{m}" for m in HTTP_METHODS]
        + [f"httpv_{v}" for v in HTTP_VERSIONS]
        + ["waterfall_index", "is_https"]
    )

    return df[feature_cols], feature_cols


def evaluate(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    med_ae = np.median(np.abs(y_true - y_pred))
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rho, _ = spearmanr(y_true, y_pred)
    # MAPE on requests > 1KB
    mask = y_true > 1000
    mape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100 if mask.sum() > 0 else float("nan")
    print(f"  {label:30s}  MAE={mae:>10,.0f}  MedAE={med_ae:>8,.0f}  RMSE={rmse:>10,.0f}  rho={rho:.4f}  MAPE(>1K)={mape:.1f}%")
    return {"mae": float(mae), "median_ae": float(med_ae), "rmse": float(rmse),
            "spearman_rho": float(rho), "mape_1k": float(mape)}


# ============================================================
# LUT Baseline
# ============================================================

def lut_baseline(train_df, eval_df):
    """Domain + resource_type lookup table baseline."""
    dt_medians = train_df.groupby(["tracker_domain", "resource_type"])["transfer_bytes"].median().to_dict()
    domain_medians = train_df.groupby("tracker_domain")["transfer_bytes"].median().to_dict()
    global_median = train_df["transfer_bytes"].median()

    def predict(row):
        key = (row["tracker_domain"], row["resource_type"])
        if key in dt_medians:
            return dt_medians[key]
        if row["tracker_domain"] in domain_medians:
            return domain_medians[row["tracker_domain"]]
        return global_median

    return eval_df.apply(predict, axis=1).values


# ============================================================
# Model training with Optuna
# ============================================================

def train_xgboost(X_train, y_train, n_trials=150):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

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
        for tr_idx, val_idx in kf.split(X_train):
            model = xgb.XGBRegressor(**params, objective="reg:squarederror",
                                      tree_method="hist", random_state=42, verbosity=0)
            model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            preds = np.clip(model.predict(X_train.iloc[val_idx]), 0, None)
            scores.append(mean_absolute_error(y_train.iloc[val_idx], preds))
        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print(f"  XGBoost best CV MAE: {study.best_value:,.0f}")

    best = xgb.XGBRegressor(**study.best_params, objective="reg:squarederror",
                             tree_method="hist", random_state=42, verbosity=0)
    best.fit(X_train, y_train)
    return best, study.best_params


def train_lightgbm(X_train, y_train, n_trials=150):
    import optuna
    import lightgbm as lgb
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
        }
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr_idx, val_idx in kf.split(X_train):
            model = lgb.LGBMRegressor(**params, random_state=42, verbosity=-1)
            model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            preds = np.clip(model.predict(X_train.iloc[val_idx]), 0, None)
            scores.append(mean_absolute_error(y_train.iloc[val_idx], preds))
        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print(f"  LightGBM best CV MAE: {study.best_value:,.0f}")

    best = lgb.LGBMRegressor(**study.best_params, random_state=42, verbosity=-1)
    best.fit(X_train, y_train)
    return best, study.best_params


def train_catboost(X_train, y_train, n_trials=100):
    import optuna
    from catboost import CatBoostRegressor
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        }
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr_idx, val_idx in kf.split(X_train):
            model = CatBoostRegressor(**params, random_seed=42, verbose=0)
            model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            preds = np.clip(model.predict(X_train.iloc[val_idx]), 0, None)
            scores.append(mean_absolute_error(y_train.iloc[val_idx], preds))
        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print(f"  CatBoost best CV MAE: {study.best_value:,.0f}")

    best = CatBoostRegressor(**study.best_params, random_seed=42, verbose=0)
    best.fit(X_train, y_train)
    return best, study.best_params


# ============================================================
# Main pipeline
# ============================================================

def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    df = load_data()
    train_df, val_df, test_df = row_level_split(df)

    # Engineer features
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

    print(f"Features: {len(feature_cols)}")
    print(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

    results = {}

    # ============================================================
    # 1. LUT Baseline
    # ============================================================
    print(f"\n{'='*60}")
    print("1. BASELINE: Domain + Resource Type LUT")
    print(f"{'='*60}")
    lut_val_preds = lut_baseline(train_df, val_df)
    lut_test_preds = lut_baseline(train_df, test_df)
    results["LUT"] = {
        "val": evaluate(y_val.values, lut_val_preds, "LUT (validation)"),
        "test": evaluate(y_test.values, lut_test_preds, "LUT (test)"),
    }

    # ============================================================
    # 2. Ridge Regression
    # ============================================================
    print(f"\n{'='*60}")
    print("2. Ridge Regression")
    print(f"{'='*60}")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train.fillna(0), y_train)
    ridge_val_preds = np.clip(ridge.predict(X_val.fillna(0)), 0, None)
    ridge_test_preds = np.clip(ridge.predict(X_test.fillna(0)), 0, None)
    results["Ridge"] = {
        "val": evaluate(y_val.values, ridge_val_preds, "Ridge (validation)"),
        "test": evaluate(y_test.values, ridge_test_preds, "Ridge (test)"),
    }

    # ============================================================
    # 3. Random Forest
    # ============================================================
    print(f"\n{'='*60}")
    print("3. Random Forest")
    print(f"{'='*60}")
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5,
                                n_jobs=-1, random_state=42)
    rf.fit(X_train.fillna(-1), y_train)
    rf_val_preds = np.clip(rf.predict(X_val.fillna(-1)), 0, None)
    rf_test_preds = np.clip(rf.predict(X_test.fillna(-1)), 0, None)
    results["RandomForest"] = {
        "val": evaluate(y_val.values, rf_val_preds, "RF (validation)"),
        "test": evaluate(y_test.values, rf_test_preds, "RF (test)"),
    }

    # ============================================================
    # 4. XGBoost (Optuna tuned)
    # ============================================================
    print(f"\n{'='*60}")
    print("4. XGBoost (Optuna, 150 trials)")
    print(f"{'='*60}")
    xgb_model, xgb_params = train_xgboost(X_train, y_train, n_trials=75)
    xgb_val_preds = np.clip(xgb_model.predict(X_val), 0, None)
    xgb_test_preds = np.clip(xgb_model.predict(X_test), 0, None)
    results["XGBoost"] = {
        "val": evaluate(y_val.values, xgb_val_preds, "XGBoost (validation)"),
        "test": evaluate(y_test.values, xgb_test_preds, "XGBoost (test)"),
        "params": xgb_params,
    }
    xgb_model.save_model(str(MODELS / "xgboost_best.json"))

    # ============================================================
    # 5. LightGBM (Optuna tuned)
    # ============================================================
    print(f"\n{'='*60}")
    print("5. LightGBM (Optuna, 150 trials)")
    print(f"{'='*60}")
    try:
        lgb_model, lgb_params = train_lightgbm(X_train, y_train, n_trials=75)
        lgb_val_preds = np.clip(lgb_model.predict(X_val), 0, None)
        lgb_test_preds = np.clip(lgb_model.predict(X_test), 0, None)
        results["LightGBM"] = {
            "val": evaluate(y_val.values, lgb_val_preds, "LightGBM (validation)"),
            "test": evaluate(y_test.values, lgb_test_preds, "LightGBM (test)"),
            "params": lgb_params,
        }
    except ImportError:
        print("  LightGBM not installed, skipping.")

    # ============================================================
    # 6. CatBoost (Optuna tuned)
    # ============================================================
    print(f"\n{'='*60}")
    print("6. CatBoost (Optuna, 100 trials)")
    print(f"{'='*60}")
    try:
        cb_model, cb_params = train_catboost(X_train, y_train, n_trials=50)
        cb_val_preds = np.clip(cb_model.predict(X_val), 0, None)
        cb_test_preds = np.clip(cb_model.predict(X_test), 0, None)
        results["CatBoost"] = {
            "val": evaluate(y_val.values, cb_val_preds, "CatBoost (validation)"),
            "test": evaluate(y_test.values, cb_test_preds, "CatBoost (test)"),
            "params": cb_params,
        }
    except ImportError:
        print("  CatBoost not installed, skipping.")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print("VALIDATION SET SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Model':20s}  {'MAE':>10s}  {'MedAE':>8s}  {'RMSE':>10s}  {'Spearman':>10s}")
    print("-" * 65)
    for name, r in results.items():
        v = r["val"]
        print(f"{name:20s}  {v['mae']:>10,.0f}  {v['median_ae']:>8,.0f}  {v['rmse']:>10,.0f}  {v['spearman_rho']:>10.4f}")

    # Best model by validation MAE
    best_name = min(results.keys(), key=lambda k: results[k]["val"]["mae"])
    print(f"\nBest model (val MAE): {best_name}")

    print(f"\n{'='*60}")
    print("TEST SET RESULTS (final, single use)")
    print(f"{'='*60}")
    print(f"\n{'Model':20s}  {'MAE':>10s}  {'MedAE':>8s}  {'RMSE':>10s}  {'Spearman':>10s}")
    print("-" * 65)
    for name, r in results.items():
        t = r["test"]
        print(f"{name:20s}  {t['mae']:>10,.0f}  {t['median_ae']:>8,.0f}  {t['rmse']:>10,.0f}  {t['spearman_rho']:>10.4f}")

    lut_mae = results["LUT"]["test"]["mae"]
    best_mae = results[best_name]["test"]["mae"]
    print(f"\nImprovement over LUT: {(1 - best_mae / lut_mae) * 100:+.1f}%")

    # ============================================================
    # Breakdown by resource type (best model on test set)
    # ============================================================
    print(f"\n{'='*60}")
    print(f"BREAKDOWN BY RESOURCE TYPE ({best_name} on test set)")
    print(f"{'='*60}")

    # Get best model test preds
    best_test_preds = {
        "LUT": lut_test_preds,
        "Ridge": ridge_test_preds,
        "RandomForest": rf_test_preds,
        "XGBoost": xgb_test_preds,
    }
    if "LightGBM" in results:
        best_test_preds["LightGBM"] = lgb_test_preds
    if "CatBoost" in results:
        best_test_preds["CatBoost"] = cb_test_preds

    bp = best_test_preds[best_name]
    lp = lut_test_preds

    print(f"\n{'Type':10s}  {'n':>6s}  {'LUT MAE':>10s}  {'Model MAE':>10s}  {'Improvement':>12s}  {'Median actual':>14s}")
    for rtype in RESOURCE_TYPES:
        mask = test_df["resource_type"].values == rtype
        if mask.sum() < 10:
            continue
        actual = y_test.values[mask]
        m_mae = mean_absolute_error(actual, bp[mask])
        l_mae = mean_absolute_error(actual, lp[mask])
        imp = (1 - m_mae / l_mae) * 100 if l_mae > 0 else 0
        med = np.median(actual)
        print(f"{rtype:10s}  {mask.sum():6d}  {l_mae:>10,.0f}  {m_mae:>10,.0f}  {imp:>+11.1f}%  {med:>14,.0f}")

    # ============================================================
    # Save results
    # ============================================================
    # Remove non-serializable items
    save_results = {}
    for name, r in results.items():
        save_results[name] = {k: v for k, v in r.items() if k in ["val", "test", "params"]}

    with open(MODELS / "per_request_results.json", "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nResults saved to {MODELS / 'per_request_results.json'}")


if __name__ == "__main__":
    main()
