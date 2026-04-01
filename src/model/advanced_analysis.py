"""
Advanced analyses for per-request transfer size prediction.

1. Calibration analysis - are predicted ranges reliable?
2. Aggregation error simulation - how accurate is the weekly total?
3. Multi-task learning - predict transfer_bytes + content_type jointly
4. Distribution shift detection - flag out-of-distribution requests

Usage:
  python src/model/advanced_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
from train_per_request import engineer_features, load_data, lut_baseline, RESOURCE_TYPES

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output"
MODELS = ROOT / "models" / "per_request"

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': 0.3, 'figure.dpi': 150,
})


def train_tweedie_model(X_train, y_train):
    """Train Tweedie model with reasonable params (skip Optuna for speed)."""
    model = xgb.XGBRegressor(
        n_estimators=250, max_depth=8, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
        tweedie_variance_power=1.5,
        objective='reg:tweedie', tree_method='hist',
        random_state=42, verbosity=0,
    )
    model.fit(X_train, y_train + 1)
    return model


def predict_tweedie(model, X):
    return np.clip(model.predict(X) - 1, 0, None)


# ============================================================
# 1. Calibration Analysis
# ============================================================

def calibration_analysis(y_true, y_pred, output_dir):
    """
    For each prediction bin, compute the actual mean.
    A well-calibrated model has predicted mean ~ actual mean in each bin.
    """
    print(f"\n{'='*60}")
    print("CALIBRATION ANALYSIS")
    print(f"{'='*60}")

    df = pd.DataFrame({'actual': y_true, 'predicted': y_pred})
    df = df[df['predicted'] > 0].copy()  # exclude zero predictions

    # Create prediction bins
    bins = [0, 100, 500, 1000, 5000, 10000, 25000, 50000, 100000, 250000, float('inf')]
    labels = ['0-100', '100-500', '500-1K', '1K-5K', '5K-10K', '10K-25K',
              '25K-50K', '50K-100K', '100K-250K', '250K+']

    df['bin'] = pd.cut(df['predicted'], bins=bins, labels=labels)

    cal_data = []
    for label in labels:
        subset = df[df['bin'] == label]
        if len(subset) < 10:
            continue
        cal_data.append({
            'bin': label,
            'n': len(subset),
            'pred_mean': subset['predicted'].mean(),
            'actual_mean': subset['actual'].mean(),
            'actual_median': subset['actual'].median(),
            'mae': mean_absolute_error(subset['actual'], subset['predicted']),
            'within_25pct': np.mean(np.abs(subset['actual'] - subset['predicted']) <= 0.25 * subset['actual'].clip(1)),
        })

    cal_df = pd.DataFrame(cal_data)

    print(f"\n{'Bin':>12s}  {'n':>6s}  {'Pred Mean':>10s}  {'Actual Mean':>12s}  {'MAE':>8s}  {'Within 25%':>10s}")
    print("-" * 65)
    for _, row in cal_df.iterrows():
        print(f"{row['bin']:>12s}  {row['n']:6.0f}  {row['pred_mean']:10,.0f}  {row['actual_mean']:12,.0f}  "
              f"{row['mae']:8,.0f}  {row['within_25pct']:10.1%}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Calibration curve
    ax = axes[0]
    valid = cal_df[cal_df['pred_mean'] > 0]
    ax.scatter(valid['pred_mean'], valid['actual_mean'], s=valid['n'] / 10, color='#3498DB', alpha=0.7, zorder=5)
    max_val = max(valid['pred_mean'].max(), valid['actual_mean'].max()) * 1.1
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, label='Perfect calibration')
    ax.set_xlabel('Predicted mean (bytes)')
    ax.set_ylabel('Actual mean (bytes)')
    ax.set_title('Calibration: Predicted vs Actual Mean per Bin')
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    for _, row in valid.iterrows():
        ax.annotate(row['bin'], (row['pred_mean'], row['actual_mean']),
                    fontsize=8, ha='left', va='bottom')

    # Accuracy by bin
    ax = axes[1]
    x = np.arange(len(cal_df))
    ax.bar(x, cal_df['within_25pct'] * 100, color='#2ECC71', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(cal_df['bin'], rotation=45, ha='right')
    ax.set_ylabel('% of predictions within 25% of actual')
    ax.set_title('Prediction Accuracy by Size Range')
    ax.axhline(50, color='#E74C3C', linestyle='--', alpha=0.5, label='50% threshold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'calibration.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved calibration.png")

    return cal_df


# ============================================================
# 2. Aggregation Error Simulation
# ============================================================

def aggregation_simulation(df, y_pred, lut_pred, n_simulations=1000, output_dir=None):
    """
    Simulate real browsing: sample N blocked requests per week,
    aggregate predictions, measure accuracy of the weekly total.
    """
    print(f"\n{'='*60}")
    print("AGGREGATION ERROR SIMULATION")
    print(f"{'='*60}")

    y_true = df['transfer_bytes'].values

    rng = np.random.RandomState(42)
    weekly_requests = [50, 100, 200, 500]

    results = []
    for n_per_week in weekly_requests:
        model_errors = []
        lut_errors = []
        model_pct_errors = []
        lut_pct_errors = []

        for _ in range(n_simulations):
            idx = rng.choice(len(y_true), size=n_per_week, replace=True)
            true_total = y_true[idx].sum()
            model_total = y_pred[idx].sum()
            lut_total = lut_pred[idx].sum()

            model_errors.append(abs(true_total - model_total))
            lut_errors.append(abs(true_total - lut_total))

            if true_total > 0:
                model_pct_errors.append(abs(true_total - model_total) / true_total * 100)
                lut_pct_errors.append(abs(true_total - lut_total) / true_total * 100)

        results.append({
            'requests_per_week': n_per_week,
            'model_mae_bytes': np.mean(model_errors),
            'lut_mae_bytes': np.mean(lut_errors),
            'model_median_pct': np.median(model_pct_errors),
            'lut_median_pct': np.median(lut_pct_errors),
            'model_p90_pct': np.percentile(model_pct_errors, 90),
            'lut_p90_pct': np.percentile(lut_pct_errors, 90),
            'model_within_10pct': np.mean(np.array(model_pct_errors) < 10),
            'lut_within_10pct': np.mean(np.array(lut_pct_errors) < 10),
        })

    results_df = pd.DataFrame(results)

    print(f"\n{'Requests/wk':>12s}  {'Model Med%':>10s}  {'LUT Med%':>10s}  {'Model <10%':>10s}  {'LUT <10%':>10s}")
    print("-" * 58)
    for _, row in results_df.iterrows():
        print(f"{row['requests_per_week']:12.0f}  {row['model_median_pct']:10.1f}%  {row['lut_median_pct']:10.1f}%  "
              f"{row['model_within_10pct']:10.1%}  {row['lut_within_10pct']:10.1%}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ax = axes[0]
    x = np.arange(len(results_df))
    width = 0.35
    ax.bar(x - width/2, results_df['model_median_pct'], width, label='Tweedie Model', color='#E74C3C', alpha=0.85)
    ax.bar(x + width/2, results_df['lut_median_pct'], width, label='LUT Baseline', color='#95A5A6', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['requests_per_week'].astype(int))
    ax.set_xlabel('Blocked requests per week')
    ax.set_ylabel('Median % error on weekly total')
    ax.set_title('Weekly Aggregate Accuracy')
    ax.legend()

    ax = axes[1]
    ax.bar(x - width/2, results_df['model_within_10pct'] * 100, width, label='Tweedie Model', color='#E74C3C', alpha=0.85)
    ax.bar(x + width/2, results_df['lut_within_10pct'] * 100, width, label='LUT Baseline', color='#95A5A6', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['requests_per_week'].astype(int))
    ax.set_xlabel('Blocked requests per week')
    ax.set_ylabel('% of weeks within 10% of true total')
    ax.set_title('Fraction of Weeks with Accurate Totals')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'aggregation_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved aggregation_accuracy.png")

    return results_df


# ============================================================
# 3. Multi-Task Learning
# ============================================================

def multitask_analysis(X_train, y_train, X_test, y_test, train_df, test_df, output_dir):
    """
    Train separate models for transfer_bytes and content_type.
    Check if sharing features between tasks improves transfer_bytes prediction.
    """
    print(f"\n{'='*60}")
    print("MULTI-TASK LEARNING")
    print(f"{'='*60}")

    # Content type as auxiliary target
    # Map content types to groups
    ct_map = {
        'application/javascript': 'javascript',
        'text/javascript': 'javascript',
        'application/x-javascript': 'javascript',
        'image/gif': 'image',
        'image/png': 'image',
        'image/jpeg': 'image',
        'image/svg+xml': 'image',
        'image/webp': 'image',
        'image/avif': 'image',
        'text/html': 'html',
        'text/plain': 'text',
        'text/css': 'css',
        'application/json': 'json',
        'video/mp4': 'video',
    }

    train_ct = train_df['content_type'].map(ct_map).fillna('other')
    test_ct = test_df['content_type'].map(ct_map).fillna('other')

    # Train content type classifier
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_ct_train = le.fit_transform(train_ct)
    y_ct_test = le.transform(test_ct)

    ct_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=7, learning_rate=0.1,
        tree_method='hist', random_state=42, verbosity=0,
        eval_metric='mlogloss',
    )
    ct_model.fit(X_train, y_ct_train)
    ct_preds = ct_model.predict(X_test)
    ct_accuracy = np.mean(ct_preds == y_ct_test)
    print(f"\n  Content type classifier accuracy: {ct_accuracy:.3f}")

    # Use content type predictions as features for transfer size
    ct_proba = ct_model.predict_proba(X_train)
    ct_proba_test = ct_model.predict_proba(X_test)

    ct_cols = [f'ct_prob_{c}' for c in le.classes_]
    X_train_mt = pd.concat([
        X_train.reset_index(drop=True),
        pd.DataFrame(ct_proba, columns=ct_cols)
    ], axis=1)
    X_test_mt = pd.concat([
        X_test.reset_index(drop=True),
        pd.DataFrame(ct_proba_test, columns=ct_cols)
    ], axis=1)

    # Train transfer size model with content type probabilities as features
    mt_model = xgb.XGBRegressor(
        n_estimators=250, max_depth=8, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
        tweedie_variance_power=1.5,
        objective='reg:tweedie', tree_method='hist',
        random_state=42, verbosity=0,
    )
    mt_model.fit(X_train_mt, y_train + 1)
    mt_preds = np.clip(mt_model.predict(X_test_mt) - 1, 0, None)

    # Compare with single-task
    st_model = train_tweedie_model(X_train, y_train)
    st_preds = predict_tweedie(st_model, X_test)

    st_mae = mean_absolute_error(y_test, st_preds)
    mt_mae = mean_absolute_error(y_test, mt_preds)

    print(f"\n  Single-task MAE: {st_mae:,.0f}")
    print(f"  Multi-task MAE:  {mt_mae:,.0f}")
    print(f"  Improvement: {(1 - mt_mae/st_mae)*100:+.1f}%")

    # Content type confusion matrix visualization
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_ct_test, ct_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(le.classes_)))
    ax.set_yticks(range(len(le.classes_)))
    ax.set_xticklabels(le.classes_, rotation=45, ha='right')
    ax.set_yticklabels(le.classes_)
    ax.set_xlabel('Predicted content type')
    ax.set_ylabel('Actual content type')
    ax.set_title(f'Content Type Classification (accuracy: {ct_accuracy:.1%})')
    plt.colorbar(im, ax=ax, label='Normalized count')

    for i in range(len(le.classes_)):
        for j in range(len(le.classes_)):
            val = cm_norm[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'content_type_confusion.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved content_type_confusion.png")

    return {
        'ct_accuracy': float(ct_accuracy),
        'single_task_mae': float(st_mae),
        'multi_task_mae': float(mt_mae),
    }


# ============================================================
# 4. Distribution Shift Detection
# ============================================================

def distribution_shift_analysis(X_train, X_test, y_test, model, output_dir):
    """
    Train a classifier to distinguish train vs test features.
    High-probability "test" samples are potential OOD requests.
    """
    print(f"\n{'='*60}")
    print("DISTRIBUTION SHIFT DETECTION")
    print(f"{'='*60}")

    # Create binary labels: 0 = train, 1 = test
    X_combined = pd.concat([
        X_train.sample(min(len(X_train), len(X_test)), random_state=42).reset_index(drop=True),
        X_test.reset_index(drop=True)
    ], ignore_index=True)
    y_domain = np.concatenate([
        np.zeros(min(len(X_train), len(X_test))),
        np.ones(len(X_test))
    ])

    # Train domain classifier
    domain_clf = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        tree_method='hist', random_state=42, verbosity=0,
        eval_metric='logloss',
    )
    domain_clf.fit(X_combined, y_domain)

    # Score test set: probability of being "from test distribution"
    # High scores = looks like test, low scores = looks like train
    # Ideally all around 0.5 (can't distinguish)
    ood_scores = domain_clf.predict_proba(X_test)[:, 1]

    # If the classifier can't distinguish, AUC should be ~0.5
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(
        np.concatenate([np.zeros(min(len(X_train), len(X_test))), np.ones(len(X_test))]),
        domain_clf.predict_proba(X_combined)[:, 1]
    )
    print(f"\n  Domain classifier AUC: {auc:.3f} (0.5 = no shift, 1.0 = total shift)")

    # Correlate OOD score with prediction error
    preds = predict_tweedie(model, X_test)
    errors = np.abs(y_test.values - preds)
    rho, p = spearmanr(ood_scores, errors)
    print(f"  OOD score vs prediction error: rho={rho:.3f} (p={p:.2e})")

    # Bin by OOD score and check error
    test_df = pd.DataFrame({
        'ood_score': ood_scores,
        'error': errors,
        'actual': y_test.values,
        'predicted': preds,
    })
    test_df['ood_bin'] = pd.qcut(test_df['ood_score'], q=5, labels=['Very low', 'Low', 'Medium', 'High', 'Very high'])

    print(f"\n  {'OOD Level':>12s}  {'n':>6s}  {'MAE':>8s}  {'Mean OOD':>10s}")
    print("  " + "-" * 42)
    for label in ['Very low', 'Low', 'Medium', 'High', 'Very high']:
        subset = test_df[test_df['ood_bin'] == label]
        print(f"  {label:>12s}  {len(subset):6d}  {subset['error'].mean():8,.0f}  {subset['ood_score'].mean():10.3f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ax = axes[0]
    ax.hist(ood_scores, bins=50, color='#3498DB', edgecolor='none', alpha=0.85)
    ax.set_xlabel('OOD Score (probability of being from test distribution)')
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution Shift Scores (AUC: {auc:.3f})')
    ax.axvline(0.5, color='#E74C3C', linestyle='--', label='No shift boundary')
    ax.legend()

    ax = axes[1]
    bin_data = test_df.groupby('ood_bin')['error'].mean()
    ax.bar(range(len(bin_data)), bin_data.values, color='#E74C3C', alpha=0.85)
    ax.set_xticks(range(len(bin_data)))
    ax.set_xticklabels(bin_data.index, rotation=15)
    ax.set_xlabel('OOD Score Level')
    ax.set_ylabel('Mean Absolute Error (bytes)')
    ax.set_title('Prediction Error by Distribution Shift Level')

    plt.tight_layout()
    plt.savefig(output_dir / 'distribution_shift.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved distribution_shift.png")

    return {'auc': float(auc), 'error_ood_correlation': float(rho)}


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT.mkdir(exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    df = load_data()
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)

    X_train, feature_cols = engineer_features(train_df, train_df)
    X_test, _ = engineer_features(train_df, test_df)

    y_train = train_df['transfer_bytes'].reset_index(drop=True)
    y_test = test_df['transfer_bytes'].reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    # Train Tweedie model
    print("Training Tweedie model...")
    model = train_tweedie_model(X_train, y_train)
    preds = predict_tweedie(model, X_test)

    # LUT baseline
    lut_preds = lut_baseline(train_df, test_df)

    # Overall metrics
    print(f"\nBaseline MAE: {mean_absolute_error(y_test, lut_preds):,.0f}")
    print(f"Tweedie MAE:  {mean_absolute_error(y_test, preds):,.0f}")

    results = {}

    # 1. Calibration
    cal_df = calibration_analysis(y_test.values, preds, OUTPUT)
    results['calibration'] = cal_df.to_dict('records')

    # 2. Aggregation simulation
    agg_df = aggregation_simulation(test_df.reset_index(drop=True), preds, lut_preds, output_dir=OUTPUT)
    results['aggregation'] = agg_df.to_dict('records')

    # 3. Multi-task
    mt_results = multitask_analysis(
        X_train, y_train, X_test, y_test,
        train_df.reset_index(drop=True), test_df.reset_index(drop=True),
        OUTPUT
    )
    results['multitask'] = mt_results

    # 4. Distribution shift
    ds_results = distribution_shift_analysis(X_train, X_test, y_test, model, OUTPUT)
    results['distribution_shift'] = ds_results

    # Save
    with open(MODELS / 'advanced_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nAll results saved to {MODELS / 'advanced_analysis_results.json'}")


if __name__ == '__main__':
    main()
