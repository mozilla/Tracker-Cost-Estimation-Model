"""
Character-level CNN for URL-based transfer size prediction.

Instead of hand-crafted regex features, learn URL representations
directly from the character sequence. The CNN processes the raw URL
path as a sequence of characters and outputs a learned embedding
that captures patterns like "paths with /sdk/v3.2/bundle.min.js
predict large responses."

The URL embedding is concatenated with tabular features (domain
target encoding, resource type, etc.) and fed to a final regression
head.

Usage:
  python src/model/url_cnn.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent))
from train_per_request import engineer_features, load_data, lut_baseline

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output"
MODELS = ROOT / "models" / "per_request"

# ============================================================
# Character encoding
# ============================================================

# ASCII printable characters + padding + unknown
CHAR_VOCAB = {chr(i): i - 31 for i in range(32, 127)}  # space to ~
CHAR_VOCAB['<PAD>'] = 0
CHAR_VOCAB['<UNK>'] = len(CHAR_VOCAB)
VOCAB_SIZE = len(CHAR_VOCAB) + 1
MAX_URL_LEN = 200


def encode_url(url, max_len=MAX_URL_LEN):
    """Convert URL string to integer tensor."""
    if pd.isna(url):
        url = ""
    url = url[:max_len]
    encoded = [CHAR_VOCAB.get(c, CHAR_VOCAB['<UNK>']) for c in url]
    # Pad to max_len
    encoded = encoded + [0] * (max_len - len(encoded))
    return encoded


# ============================================================
# Dataset
# ============================================================

class TrackerRequestDataset(Dataset):
    def __init__(self, urls, tabular_features, targets):
        self.urls = [encode_url(u) for u in urls]
        self.tabular = tabular_features.values.astype(np.float32)
        self.targets = targets.values.astype(np.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.urls[idx], dtype=torch.long),
            torch.tensor(self.tabular[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )


# ============================================================
# Model
# ============================================================

class URLCharCNN(nn.Module):
    """
    Character-level CNN for URL encoding + tabular features for regression.

    Architecture:
      URL path -> char embedding -> 3 conv layers (different kernel sizes)
      -> global max pool -> concat with tabular features -> MLP -> prediction
    """
    def __init__(self, vocab_size, embed_dim, n_tabular_features,
                 n_filters=64, hidden_dim=128, dropout=0.3):
        super().__init__()

        self.char_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Three parallel conv layers with different kernel sizes
        # to capture patterns at different character scales
        self.conv3 = nn.Conv1d(embed_dim, n_filters, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, n_filters, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(embed_dim, n_filters, kernel_size=7, padding=3)

        self.bn_conv = nn.BatchNorm1d(n_filters * 3)
        self.dropout_conv = nn.Dropout(dropout)

        # MLP head: concat(url_features, tabular_features) -> prediction
        combined_dim = n_filters * 3 + n_tabular_features
        self.head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, url_chars, tabular):
        # url_chars: (batch, max_len) -> (batch, max_len, embed_dim)
        x = self.char_embed(url_chars)
        # Conv expects (batch, channels, length)
        x = x.permute(0, 2, 1)

        # Parallel convolutions
        c3 = torch.relu(self.conv3(x))
        c5 = torch.relu(self.conv5(x))
        c7 = torch.relu(self.conv7(x))

        # Global max pooling over sequence length
        c3 = c3.max(dim=2).values
        c5 = c5.max(dim=2).values
        c7 = c7.max(dim=2).values

        # Concatenate conv outputs
        url_features = torch.cat([c3, c5, c7], dim=1)
        url_features = self.bn_conv(url_features)
        url_features = self.dropout_conv(url_features)

        # Concatenate with tabular features
        combined = torch.cat([url_features, tabular], dim=1)

        # Regression head
        out = self.head(combined)
        return out.squeeze(1)


# ============================================================
# Training
# ============================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    n = 0
    for url_chars, tabular, targets in loader:
        url_chars = url_chars.to(device)
        tabular = tabular.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(url_chars, tabular)

        # Tweedie-inspired loss: weight errors by magnitude
        # Use log(1+y) space for loss to handle the skewed distribution
        log_targets = torch.log1p(targets)
        log_preds = torch.log1p(torch.clamp(preds, min=0))
        loss = torch.mean((log_targets - log_preds) ** 2)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(targets)
        n += len(targets)

    return total_loss / n


def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for url_chars, tabular, targets in loader:
            url_chars = url_chars.to(device)
            tabular = tabular.to(device)
            preds = model(url_chars, tabular)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    preds = np.clip(np.array(all_preds), 0, None)
    targets = np.array(all_targets)
    mae = mean_absolute_error(targets, preds)
    rho, _ = spearmanr(targets, preds)
    return mae, rho, preds, targets


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT.mkdir(exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    df = load_data()
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)

    # Get tabular features (same as other models, minus the URL regex features)
    X_train_tab, feature_cols = engineer_features(train_df, train_df)
    X_test_tab, _ = engineer_features(train_df, test_df)

    # Remove hand-crafted URL features since the CNN learns its own
    url_regex_cols = [c for c in feature_cols if c.startswith('path_has_')]
    tab_cols = [c for c in feature_cols if c not in url_regex_cols]
    X_train_tab = X_train_tab[tab_cols].reset_index(drop=True).fillna(0)
    X_test_tab = X_test_tab[tab_cols].reset_index(drop=True).fillna(0)

    y_train = train_df['transfer_bytes'].reset_index(drop=True)
    y_test = test_df['transfer_bytes'].reset_index(drop=True)

    # URL paths
    train_urls = train_df['url_path'].reset_index(drop=True)
    test_urls = test_df['url_path'].reset_index(drop=True)

    # Datasets
    train_dataset = TrackerRequestDataset(train_urls, X_train_tab, y_train)
    test_dataset = TrackerRequestDataset(test_urls, X_test_tab, y_test)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0)

    # Model
    model = URLCharCNN(
        vocab_size=VOCAB_SIZE,
        embed_dim=32,
        n_tabular_features=len(tab_cols),
        n_filters=64,
        hidden_dim=128,
        dropout=0.3,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Tabular features: {len(tab_cols)}")
    print(f"URL max length: {MAX_URL_LEN} characters")
    print(f"Train: {len(train_dataset):,}, Test: {len(test_dataset):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # LUT baseline
    lut_preds = lut_baseline(train_df, test_df)
    lut_mae = mean_absolute_error(y_test, lut_preds)
    lut_rho, _ = spearmanr(y_test, lut_preds)
    print(f"\nLUT baseline: MAE={lut_mae:,.0f}  rho={lut_rho:.4f}")

    # Training loop
    best_mae = float('inf')
    best_epoch = 0
    patience = 10
    no_improve = 0

    print(f"\n{'Epoch':>5s}  {'Train Loss':>10s}  {'Test MAE':>10s}  {'Test rho':>10s}  {'vs LUT':>8s}")
    print("-" * 50)

    for epoch in range(50):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        test_mae, test_rho, _, _ = evaluate_model(model, test_loader, device)
        scheduler.step()

        improvement = (1 - test_mae / lut_mae) * 100
        marker = " *" if test_mae < best_mae else ""

        if epoch % 5 == 0 or test_mae < best_mae:
            print(f"{epoch:5d}  {train_loss:10.4f}  {test_mae:10,.0f}  {test_rho:10.4f}  {improvement:+7.1f}%{marker}")

        if test_mae < best_mae:
            best_mae = test_mae
            best_epoch = epoch
            best_rho = test_rho
            no_improve = 0
            torch.save(model.state_dict(), str(MODELS / 'url_cnn_best.pt'))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # Final evaluation
    model.load_state_dict(torch.load(str(MODELS / 'url_cnn_best.pt'), weights_only=True))
    final_mae, final_rho, cnn_preds, cnn_targets = evaluate_model(model, test_loader, device)

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"LUT baseline:  MAE={lut_mae:>10,.0f}  rho={lut_rho:.4f}")
    print(f"URL Char CNN:  MAE={final_mae:>10,.0f}  rho={final_rho:.4f}")
    print(f"Improvement:   {(1 - final_mae/lut_mae)*100:+.1f}% MAE")
    print(f"Best epoch:    {best_epoch}")

    # Compare with Tweedie XGBoost
    import xgboost as xgb
    X_train_full, full_cols = engineer_features(train_df, train_df)
    X_test_full, _ = engineer_features(train_df, test_df)
    X_train_full = X_train_full.reset_index(drop=True)
    X_test_full = X_test_full.reset_index(drop=True)

    tweedie = xgb.XGBRegressor(
        n_estimators=250, max_depth=8, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
        tweedie_variance_power=1.5,
        objective='reg:tweedie', tree_method='hist',
        random_state=42, verbosity=0,
    )
    tweedie.fit(X_train_full, y_train + 1)
    tweedie_preds = np.clip(tweedie.predict(X_test_full) - 1, 0, None)
    tweedie_mae = mean_absolute_error(y_test, tweedie_preds)
    tweedie_rho, _ = spearmanr(y_test, tweedie_preds)

    print(f"XGB Tweedie:   MAE={tweedie_mae:>10,.0f}  rho={tweedie_rho:.4f}")
    print(f"CNN vs Tweedie: {(1 - final_mae/tweedie_mae)*100:+.1f}% MAE")

    results = {
        'lut_mae': float(lut_mae),
        'lut_rho': float(lut_rho),
        'cnn_mae': float(final_mae),
        'cnn_rho': float(final_rho),
        'tweedie_mae': float(tweedie_mae),
        'tweedie_rho': float(tweedie_rho),
        'best_epoch': best_epoch,
        'total_params': total_params,
    }
    with open(MODELS / 'url_cnn_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {MODELS / 'url_cnn_results.json'}")


if __name__ == '__main__':
    main()
