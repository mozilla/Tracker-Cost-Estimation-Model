"""
Generate all figures for the paper.
Publication-quality, IMC-style (CDFs, clean bar charts, minimal decoration).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from url_embeddings import URLEmbedder
from train_multi_target import engineer_features, TARGETS

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models" / "per_request"
FIGDIR = ROOT / "figures"
FIGDIR.mkdir(exist_ok=True)

# Consistent style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

COLORS = {
    'model': '#2166ac',
    'lut': '#b2182b',
    'path_lut': '#ef8a62',
    'domain_lut': '#d6604d',
    'global': '#999999',
    'script': '#2166ac',
    'image': '#67a9cf',
    'html': '#d1e5f0',
    'other': '#f4a582',
    'text': '#fddbc7',
    'css': '#92c5de',
}


def load_data():
    print("Loading data...")
    df = pd.read_csv(ROOT / 'data' / 'raw' / 'per_request_1pct.csv', low_memory=False)
    np.random.seed(42)
    n = len(df)
    idx = np.random.permutation(n)
    train_df = df.iloc[idx[:int(0.7 * n)]].reset_index(drop=True)
    val_df = df.iloc[idx[int(0.7 * n):int(0.85 * n)]].reset_index(drop=True)
    test_df = df.iloc[idx[int(0.85 * n):]].reset_index(drop=True)

    embedder = URLEmbedder(n_components=50)
    embedder.fit(train_df['url_path'].fillna(''))
    embed_train = embedder.transform(train_df['url_path'].fillna(''))
    embed_test = embedder.transform(test_df['url_path'].fillna(''))

    X_train = engineer_features(train_df, train_df, embed_train)
    X_test = engineer_features(test_df, train_df, embed_test)

    return df, train_df, val_df, test_df, X_train, X_test, embedder


def fig1_target_distribution(df):
    """CDF of transfer_bytes — shows zero-inflation and heavy tail."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))

    y = df['transfer_bytes'].clip(lower=0).values

    # Left: CDF with log x-axis
    sorted_y = np.sort(y[y > 0])
    cdf = np.arange(1, len(sorted_y) + 1) / len(y)  # includes zeros in denominator
    ax1.plot(sorted_y, cdf, color=COLORS['model'], linewidth=1.5)
    ax1.axhline(y=1 - (y == 0).mean(), color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax1.text(50, 1 - (y == 0).mean() + 0.02, f'{(y==0).mean()*100:.0f}% zeros',
             fontsize=8, color='gray')
    ax1.set_xscale('log')
    ax1.set_xlabel('Transfer size (bytes)')
    ax1.set_ylabel('CDF')
    ax1.set_xlim(1, 1e7)
    ax1.set_ylim(0, 1.02)
    ax1.set_title('(a) Transfer size CDF')
    ax1.grid(True, alpha=0.3)

    # Right: histogram of log(1+y) to show bimodality
    log_y = np.log10(1 + y)
    ax2.hist(log_y, bins=100, density=True, color=COLORS['model'], alpha=0.7, edgecolor='none')
    ax2.axvline(x=np.log10(1), color='gray', linestyle='--', linewidth=0.8)
    ax2.text(np.log10(1) + 0.1, ax2.get_ylim()[1] * 0.9, 'zeros + beacons',
             fontsize=7, color='gray')
    ax2.axvline(x=np.log10(90000), color='gray', linestyle='--', linewidth=0.8)
    ax2.text(np.log10(90000) - 0.8, ax2.get_ylim()[1] * 0.9, 'JS bundles',
             fontsize=7, color='gray')
    ax2.set_xlabel('log₁₀(1 + transfer_bytes)')
    ax2.set_ylabel('Density')
    ax2.set_title('(b) Log-scale distribution')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGDIR / 'fig1_target_distribution.pdf')
    fig.savefig(FIGDIR / 'fig1_target_distribution.png')
    plt.close()
    print("  fig1_target_distribution")


def fig2_baseline_ladder(train_df, test_df, X_test):
    """Bar chart: MAE for each baseline approach."""
    y_test = test_df['transfer_bytes'].clip(lower=0).values

    # Global
    g = train_df['transfer_bytes'].median()
    mae_global = mean_absolute_error(y_test, np.full(len(y_test), g))

    # Domain
    d_med = train_df.groupby('tracker_domain')['transfer_bytes'].median()
    d_preds = test_df['tracker_domain'].map(d_med).fillna(g).values
    mae_domain = mean_absolute_error(y_test, d_preds)

    # Domain+type
    dt_med = train_df.groupby(['tracker_domain', 'resource_type'])['transfer_bytes'].median()
    d_med2 = train_df.groupby('tracker_domain')['transfer_bytes'].median()
    dt_keys = pd.MultiIndex.from_frame(test_df[['tracker_domain', 'resource_type']])
    dt_preds = pd.Series([dt_med.get(k, np.nan) for k in dt_keys], index=test_df.index)
    mask = dt_preds.isna()
    dt_preds[mask] = test_df.loc[mask, 'tracker_domain'].map(d_med2)
    dt_preds = dt_preds.fillna(g).values
    mae_dt = mean_absolute_error(y_test, dt_preds)

    # Path LUT
    path_med = train_df.groupby(['tracker_domain', 'url_path'])['transfer_bytes'].median()
    pk = list(zip(test_df['tracker_domain'], test_df['url_path']))
    path_preds = pd.Series([path_med.get(k, np.nan) for k in pk], index=test_df.index)
    miss = path_preds.isna()
    path_preds[miss] = pd.Series([dt_med.get(k, np.nan) for k in
                                   pd.MultiIndex.from_frame(test_df.loc[miss, ['tracker_domain', 'resource_type']])],
                                  index=test_df[miss].index)
    miss2 = path_preds.isna()
    path_preds[miss2] = test_df.loc[miss2, 'tracker_domain'].map(d_med2)
    path_preds = path_preds.fillna(g).values
    mae_path = mean_absolute_error(y_test, path_preds)

    # Model
    model = xgb.XGBRegressor()
    model.load_model(str(MODELS / 'xgb_transfer_bytes.json'))
    model_preds = np.clip(model.predict(X_test) - 1, 0, None)
    mae_model = mean_absolute_error(y_test, model_preds)

    approaches = ['Global\nmedian', 'Domain\nLUT', 'Domain+type\nLUT', 'Path\nLUT', 'XGBoost\nTweedie']
    maes = [mae_global, mae_domain, mae_dt, mae_path, mae_model]
    colors = [COLORS['global'], COLORS['domain_lut'], COLORS['lut'], COLORS['path_lut'], COLORS['model']]

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(approaches, maes, color=colors, edgecolor='white', width=0.65)

    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f'{mae:,.0f}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('MAE (bytes)')
    ax.set_title('Transfer size prediction: baseline hierarchy')
    ax.set_ylim(0, max(maes) * 1.15)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIGDIR / 'fig2_baseline_ladder.pdf')
    fig.savefig(FIGDIR / 'fig2_baseline_ladder.png')
    plt.close()
    print("  fig2_baseline_ladder")


def fig3_aggregation_cdf(train_df, test_df, X_test):
    """CDF of weekly aggregation error: model vs LUT."""
    y_test = test_df['transfer_bytes'].clip(lower=0).values

    # LUT predictions
    g = train_df['transfer_bytes'].median()
    dt_med = train_df.groupby(['tracker_domain', 'resource_type'])['transfer_bytes'].median()
    d_med = train_df.groupby('tracker_domain')['transfer_bytes'].median()
    dt_keys = pd.MultiIndex.from_frame(test_df[['tracker_domain', 'resource_type']])
    lut_preds = pd.Series([dt_med.get(k, np.nan) for k in dt_keys], index=test_df.index)
    mask = lut_preds.isna()
    lut_preds[mask] = test_df.loc[mask, 'tracker_domain'].map(d_med)
    lut_preds = lut_preds.fillna(g).values

    # Model predictions
    model = xgb.XGBRegressor()
    model.load_model(str(MODELS / 'xgb_transfer_bytes.json'))
    model_preds = np.clip(model.predict(X_test) - 1, 0, None)

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    for N, ls in [(100, '--'), (200, '-'), (500, ':')]:
        model_errs = []
        lut_errs = []
        np.random.seed(42)
        for _ in range(2000):
            idx = np.random.choice(len(y_test), size=N, replace=True)
            true_sum = y_test[idx].sum()
            if true_sum == 0:
                continue
            model_errs.append(abs(model_preds[idx].sum() - true_sum) / true_sum * 100)
            lut_errs.append(abs(lut_preds[idx].sum() - true_sum) / true_sum * 100)

        model_sorted = np.sort(model_errs)
        lut_sorted = np.sort(lut_errs)
        cdf = np.arange(1, len(model_sorted) + 1) / len(model_sorted)

        ax.plot(model_sorted, cdf, color=COLORS['model'], linestyle=ls,
                linewidth=1.5, label=f'Model (N={N})')
        ax.plot(lut_sorted, cdf, color=COLORS['lut'], linestyle=ls,
                linewidth=1.5, label=f'LUT (N={N})')

    ax.axvline(x=10, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.text(10.5, 0.05, '10% error', fontsize=7, color='gray')

    ax.set_xlabel('Weekly aggregate error (%)')
    ax.set_ylabel('CDF')
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 1.02)
    ax.legend(loc='lower right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIGDIR / 'fig3_aggregation_cdf.pdf')
    fig.savefig(FIGDIR / 'fig3_aggregation_cdf.png')
    plt.close()
    print("  fig3_aggregation_cdf")


def fig4_per_resource_type(train_df, test_df, X_test):
    """Grouped bar chart: MAE by resource type, model vs LUT."""
    y_test = test_df['transfer_bytes'].clip(lower=0)

    g = train_df['transfer_bytes'].median()
    dt_med = train_df.groupby(['tracker_domain', 'resource_type'])['transfer_bytes'].median()
    d_med = train_df.groupby('tracker_domain')['transfer_bytes'].median()
    dt_keys = pd.MultiIndex.from_frame(test_df[['tracker_domain', 'resource_type']])
    lut_preds = pd.Series([dt_med.get(k, np.nan) for k in dt_keys], index=test_df.index)
    mask = lut_preds.isna()
    lut_preds[mask] = test_df.loc[mask, 'tracker_domain'].map(d_med)
    lut_preds = lut_preds.fillna(g).values

    model = xgb.XGBRegressor()
    model.load_model(str(MODELS / 'xgb_transfer_bytes.json'))
    model_preds = np.clip(model.predict(X_test) - 1, 0, None)

    types = ['script', 'css', 'html', 'image', 'other', 'text']
    lut_maes = []
    mod_maes = []
    counts = []
    for rt in types:
        m = test_df['resource_type'] == rt
        if m.sum() < 100:
            continue
        lut_maes.append(mean_absolute_error(y_test[m], lut_preds[m]))
        mod_maes.append(mean_absolute_error(y_test[m], model_preds[m]))
        counts.append(m.sum())

    x = np.arange(len(types))
    w = 0.35

    fig, ax = plt.subplots(figsize=(5.5, 3))
    bars1 = ax.bar(x - w/2, lut_maes, w, label='Domain+type LUT', color=COLORS['lut'], edgecolor='white')
    bars2 = ax.bar(x + w/2, mod_maes, w, label='XGBoost Tweedie', color=COLORS['model'], edgecolor='white')

    # Improvement annotations
    for i, (lm, mm) in enumerate(zip(lut_maes, mod_maes)):
        if lm > 0:
            improv = (1 - mm / lm) * 100
            if abs(improv) > 3:
                y_pos = max(lm, mm) + max(lut_maes) * 0.03
                ax.text(x[i], y_pos, f'{improv:+.0f}%', ha='center', fontsize=7,
                        color=COLORS['model'] if improv > 0 else COLORS['lut'])

    ax.set_ylabel('MAE (bytes)')
    ax.set_xticks(x)
    labels = [f'{t}\n(n={c:,})' for t, c in zip(types, counts)]
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(loc='upper right', fontsize=8)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('MAE by resource type')

    plt.tight_layout()
    fig.savefig(FIGDIR / 'fig4_per_resource_type.pdf')
    fig.savefig(FIGDIR / 'fig4_per_resource_type.png')
    plt.close()
    print("  fig4_per_resource_type")


def fig5_loss_function(train_df, test_df, X_train, X_test):
    """Bar chart: loss function ablation."""
    y_test = test_df['transfer_bytes'].clip(lower=0).values

    # From our experiment results
    losses = ['Squared\nerror', 'Tweedie\np=1.2', 'Tweedie\np=1.5', 'Tweedie\np=1.8']
    maes = [4527, 3486, 3466, 3597]
    rhos = [0.738, 0.937, 0.945, 0.949]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.8))

    # MAE bars
    colors = ['#999999', '#67a9cf', COLORS['model'], '#67a9cf']
    bars = ax1.bar(losses, maes, color=colors, edgecolor='white', width=0.55)
    for bar, mae in zip(bars, maes):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f'{mae:,}', ha='center', va='bottom', fontsize=8)
    ax1.set_ylabel('MAE (bytes)')
    ax1.set_title('(a) Mean Absolute Error')
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax1.set_ylim(0, 5200)
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Spearman bars
    bars2 = ax2.bar(losses, rhos, color=colors, edgecolor='white', width=0.55)
    for bar, rho in zip(bars2, rhos):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{rho:.3f}', ha='center', va='bottom', fontsize=8)
    ax2.set_ylabel('Spearman ρ')
    ax2.set_title('(b) Ranking quality')
    ax2.set_ylim(0.6, 1.0)
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIGDIR / 'fig5_loss_ablation.pdf')
    fig.savefig(FIGDIR / 'fig5_loss_ablation.png')
    plt.close()
    print("  fig5_loss_ablation")


def fig6_pred_vs_actual(test_df, X_test):
    """Scatter: predicted vs actual (log scale), model vs LUT."""
    y_test = test_df['transfer_bytes'].clip(lower=0).values

    model = xgb.XGBRegressor()
    model.load_model(str(MODELS / 'xgb_transfer_bytes.json'))
    model_preds = np.clip(model.predict(X_test) - 1, 0, None)

    fig, ax = plt.subplots(figsize=(4, 3.5))

    # Subsample for plotting
    np.random.seed(42)
    sample_idx = np.random.choice(len(y_test), size=min(20000, len(y_test)), replace=False)
    ys = y_test[sample_idx]
    ps = model_preds[sample_idx]

    # Only plot nonzero
    mask = (ys > 0) & (ps > 0)
    ax.scatter(ys[mask], ps[mask], alpha=0.03, s=2, color=COLORS['model'], rasterized=True)

    # Diagonal
    lims = [1, 1e7]
    ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.5)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Actual transfer size (bytes)')
    ax.set_ylabel('Predicted transfer size (bytes)')
    ax.set_xlim(1, 1e7)
    ax.set_ylim(1, 1e7)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIGDIR / 'fig6_pred_vs_actual.pdf')
    fig.savefig(FIGDIR / 'fig6_pred_vs_actual.png')
    plt.close()
    print("  fig6_pred_vs_actual")


def fig7_within_domain_variance(df):
    """CDF of within-domain CV for each target."""
    fig, ax = plt.subplots(figsize=(4.5, 3))

    targets_to_plot = {
        'transfer_bytes': ('Transfer size', COLORS['model']),
        'download_ms': ('Download time', '#67a9cf'),
        'load_ms': ('Load time', COLORS['lut']),
        'ttfb_ms': ('TTFB', COLORS['path_lut']),
    }

    for col, (label, color) in targets_to_plot.items():
        grp = df.groupby('tracker_domain')[col].agg(['mean', 'std', 'count'])
        grp = grp[grp['count'] >= 10]
        cv = (grp['std'] / grp['mean'].clip(lower=1)).values
        cv_sorted = np.sort(cv)
        cdf = np.arange(1, len(cv_sorted) + 1) / len(cv_sorted)
        ax.plot(cv_sorted, cdf, label=label, color=color, linewidth=1.5)

    ax.axvline(x=1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Within-domain coefficient of variation')
    ax.set_ylabel('CDF of domains')
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1.02)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIGDIR / 'fig7_within_domain_cv.pdf')
    fig.savefig(FIGDIR / 'fig7_within_domain_cv.png')
    plt.close()
    print("  fig7_within_domain_cv")


def fig8_feature_ablation():
    """Bar chart: feature ablation results."""
    configs = ['Regex\nonly', 'TF-IDF\nonly', 'Both']
    maes = [4251, 3548, 3466]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    colors = ['#999999', '#67a9cf', COLORS['model']]
    bars = ax.bar(configs, maes, color=colors, edgecolor='white', width=0.5)

    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f'{mae:,}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('MAE (bytes)')
    ax.set_title('URL content feature ablation')
    ax.set_ylim(0, 4800)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIGDIR / 'fig8_feature_ablation.pdf')
    fig.savefig(FIGDIR / 'fig8_feature_ablation.png')
    plt.close()
    print("  fig8_feature_ablation")


def fig9_path_decomposition():
    """Grouped bar: model vs path LUT on seen vs unseen paths."""
    subsets = ['Path seen\n(91.6%)', 'Path unseen\n(8.4%)', 'Overall']
    lut_maes = [1448, 29491, 3797]
    model_maes = [1346, 26655, 3466]

    x = np.arange(len(subsets))
    w = 0.32

    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.bar(x - w/2, lut_maes, w, label='Path LUT', color=COLORS['path_lut'], edgecolor='white')
    ax.bar(x + w/2, model_maes, w, label='XGBoost Tweedie', color=COLORS['model'], edgecolor='white')

    for i, (lm, mm) in enumerate(zip(lut_maes, model_maes)):
        y_pos = max(lm, mm) + max(lut_maes) * 0.03
        improv = (1 - mm / lm) * 100
        ax.text(x[i], y_pos, f'{improv:+.1f}%', ha='center', fontsize=8, color=COLORS['model'])

    ax.set_ylabel('MAE (bytes)')
    ax.set_xticks(x)
    ax.set_xticklabels(subsets, fontsize=9)
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Model vs. path lookup by coverage')

    plt.tight_layout()
    fig.savefig(FIGDIR / 'fig9_path_decomposition.pdf')
    fig.savefig(FIGDIR / 'fig9_path_decomposition.png')
    plt.close()
    print("  fig9_path_decomposition")


def fig10_multi_target_aggregation():
    """Multi-target aggregation: model vs LUT at N=200."""
    targets = ['transfer\n_bytes', 'download\n_ms', 'load\n_ms', 'ttfb\n_ms']
    model_err = [6.0, 16.6, 5.6, 4.0]
    lut_err = [21.9, 37.1, 14.2, 13.8]

    x = np.arange(len(targets))
    w = 0.32

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(x - w/2, lut_err, w, label='Domain+type LUT', color=COLORS['lut'], edgecolor='white')
    ax.bar(x + w/2, model_err, w, label='XGBoost Tweedie', color=COLORS['model'], edgecolor='white')

    ax.axhline(y=10, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.text(3.6, 10.5, '10% threshold', fontsize=7, color='gray')

    ax.set_ylabel('Median weekly error (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(targets, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Weekly aggregation error (N=200 requests)')

    plt.tight_layout()
    fig.savefig(FIGDIR / 'fig10_multi_target_agg.pdf')
    fig.savefig(FIGDIR / 'fig10_multi_target_agg.png')
    plt.close()
    print("  fig10_multi_target_agg")


def fig11_domain_example(df):
    """Example: googletagmanager.com URL path vs transfer size."""
    gtm = df[df['tracker_domain'] == 'www.googletagmanager.com'].copy()
    if len(gtm) == 0:
        gtm = df[df['tracker_domain'].str.contains('googletagmanager', na=False)].copy()
    if len(gtm) == 0:
        print("  fig11_domain_example SKIPPED (no GTM data)")
        return

    gtm['path_short'] = gtm['url_path'].fillna('').str[:30]
    top_paths = gtm.groupby('path_short')['transfer_bytes'].agg(['median', 'count', 'std']).reset_index()
    top_paths = top_paths[top_paths['count'] >= 20].nlargest(8, 'count')

    fig, ax = plt.subplots(figsize=(5.5, 3))
    bars = ax.barh(range(len(top_paths)), top_paths['median'].values,
                   xerr=top_paths['std'].clip(upper=top_paths['median'] * 2).values,
                   color=COLORS['model'], edgecolor='white', height=0.6,
                   error_kw={'linewidth': 0.8, 'capsize': 2})
    ax.set_yticks(range(len(top_paths)))
    labels = [p[:25] + '...' if len(p) > 25 else p for p in top_paths['path_short'].values]
    ax.set_yticklabels(labels, fontsize=7, fontfamily='monospace')
    ax.set_xlabel('Median transfer size (bytes)')
    ax.set_title('googletagmanager.com: same domain, different costs')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(FIGDIR / 'fig11_domain_example.pdf')
    fig.savefig(FIGDIR / 'fig11_domain_example.png')
    plt.close()
    print("  fig11_domain_example")


def main():
    df, train_df, val_df, test_df, X_train, X_test, embedder = load_data()

    print("\nGenerating figures...")
    fig1_target_distribution(df)
    fig2_baseline_ladder(train_df, test_df, X_test)
    fig3_aggregation_cdf(train_df, test_df, X_test)
    fig4_per_resource_type(train_df, test_df, X_test)
    fig5_loss_function(train_df, test_df, X_train, X_test)
    fig6_pred_vs_actual(test_df, X_test)
    fig7_within_domain_variance(df)
    fig8_feature_ablation()
    fig9_path_decomposition()
    fig10_multi_target_aggregation()
    fig11_domain_example(df)

    print(f"\nAll figures saved to {FIGDIR}/")
    print(f"Files: {sorted(f.name for f in FIGDIR.glob('*.png'))}")


if __name__ == '__main__':
    main()
