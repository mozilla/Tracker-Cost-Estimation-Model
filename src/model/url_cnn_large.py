"""
Large-scale character-level CNN for URL embedding learning.

Scaled-up version of url_cnn.py designed for 35M+ rows.
Trains on transfer_bytes, then the learned 768-dim URL embeddings
can be extracted and used as features for XGBoost across all targets.

Architecture:
  - 64-dim character embeddings (2x original)
  - 256 filters per conv width (4x original)
  - 4 parallel conv widths: 3, 5, 7, 11 chars
  - Second conv block with residual connection
  - 512-dim MLP head (4x original)
  - ~2.5M parameters (35x original)

Usage:
  # Train and save embeddings:
  python src/model/url_cnn_large.py --data data/raw/per_request_10pct/ --epochs 30

  # Extract embeddings only (from saved model):
  python src/model/url_cnn_large.py --extract --model models/per_request/url_cnn_large_best.pt
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from pathlib import Path
import sys
import json
import glob
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output"
MODELS = ROOT / "models" / "per_request"

# ============================================================
# Character encoding
# ============================================================

CHAR_VOCAB = {chr(i): i - 31 for i in range(32, 127)}
CHAR_VOCAB['<PAD>'] = 0
CHAR_VOCAB['<UNK>'] = len(CHAR_VOCAB)
VOCAB_SIZE = len(CHAR_VOCAB) + 1
MAX_URL_LEN = 300  # increased from 200 to capture longer tracker URLs


def encode_url(url, max_len=MAX_URL_LEN):
    """Convert URL string to integer tensor."""
    if pd.isna(url):
        url = ""
    url = str(url)[:max_len]
    encoded = [CHAR_VOCAB.get(c, CHAR_VOCAB['<UNK>']) for c in url]
    encoded = encoded + [0] * (max_len - len(encoded))
    return encoded


# ============================================================
# Dataset — supports sharded CSVs for large data
# ============================================================

class TrackerRequestDataset(Dataset):
    """In-memory dataset for moderate sizes (< 5M rows)."""

    def __init__(self, urls, tabular_features, targets):
        self.urls = [encode_url(u) for u in urls]
        self.tabular = tabular_features.astype(np.float32)
        self.targets = targets.astype(np.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.urls[idx], dtype=torch.long),
            torch.tensor(self.tabular[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )


class ShardedTrackerDataset(Dataset):
    """Memory-mapped dataset that loads sharded CSVs on demand.

    For 35M+ rows that don't fit in RAM. Each shard is loaded
    when first accessed and cached with LRU eviction.
    """

    def __init__(self, shard_dir, feature_cols, target_col, url_col='url_path',
                 max_cached_shards=5):
        self.shard_dir = Path(shard_dir)
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.url_col = url_col
        self.max_cached_shards = max_cached_shards

        # Find all shard files
        self.shard_paths = sorted(
            glob.glob(str(self.shard_dir / "*.csv")) +
            glob.glob(str(self.shard_dir / "*.csv.gz"))
        )
        if not self.shard_paths:
            raise FileNotFoundError(f"No CSV shards found in {shard_dir}")

        # Count rows per shard (read just the index)
        self.shard_sizes = []
        self.cumulative_sizes = [0]
        for path in self.shard_paths:
            # Quick row count without loading data
            n = sum(1 for _ in open(path)) - 1 if not path.endswith('.gz') else len(pd.read_csv(path))
            self.shard_sizes.append(n)
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + n)

        self.total_rows = self.cumulative_sizes[-1]
        self._cache = {}
        self._cache_order = []

    def __len__(self):
        return self.total_rows

    def _get_shard_idx(self, global_idx):
        """Map global index to (shard_index, local_index)."""
        for i, cumsize in enumerate(self.cumulative_sizes[1:]):
            if global_idx < cumsize:
                local_idx = global_idx - self.cumulative_sizes[i]
                return i, local_idx
        raise IndexError(f"Index {global_idx} out of range")

    def _load_shard(self, shard_idx):
        """Load and cache a shard."""
        if shard_idx in self._cache:
            return self._cache[shard_idx]

        # Evict oldest if cache full
        while len(self._cache) >= self.max_cached_shards:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]

        df = pd.read_csv(self.shard_paths[shard_idx])
        shard_data = {
            'urls': [encode_url(u) for u in df[self.url_col]],
            'tabular': df[self.feature_cols].fillna(0).values.astype(np.float32),
            'targets': df[self.target_col].fillna(0).values.astype(np.float32),
        }
        self._cache[shard_idx] = shard_data
        self._cache_order.append(shard_idx)
        return shard_data

    def __getitem__(self, idx):
        shard_idx, local_idx = self._get_shard_idx(idx)
        shard = self._load_shard(shard_idx)
        return (
            torch.tensor(shard['urls'][local_idx], dtype=torch.long),
            torch.tensor(shard['tabular'][local_idx], dtype=torch.float32),
            torch.tensor(shard['targets'][local_idx], dtype=torch.float32),
        )


# ============================================================
# Model
# ============================================================

class URLCharCNNLarge(nn.Module):
    """
    Scaled-up character-level CNN for URL embedding learning.

    Architecture:
      URL path → 64-dim char embedding
      → 4 parallel conv layers (kernel 3/5/7/11, 256 filters each)
      → global max pool → 1024-dim
      → second conv block with residual (256 filters, kernel 3)
      → batch norm + dropout
      → concat with tabular features
      → MLP: (1024 + tabular) → 512 → 256 → 1

    The 1024-dim vector after the first pool is the "URL embedding"
    that gets extracted for use as XGBoost features.
    """

    def __init__(self, vocab_size, embed_dim=64, n_tabular_features=25,
                 n_filters=256, hidden_dim=512, dropout=0.3):
        super().__init__()

        self.char_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Four parallel conv layers — wider kernel range than original
        self.conv3 = nn.Conv1d(embed_dim, n_filters, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, n_filters, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(embed_dim, n_filters, kernel_size=7, padding=3)
        self.conv11 = nn.Conv1d(embed_dim, n_filters, kernel_size=11, padding=5)

        self.n_conv_out = n_filters * 4  # 1024

        # Second conv block: operates on the concatenated conv outputs
        # before global pooling, to learn cross-scale interactions
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(self.n_conv_out, self.n_conv_out, kernel_size=3, padding=1, groups=4),
            nn.ReLU(),
            nn.Conv1d(self.n_conv_out, self.n_conv_out, kernel_size=1),  # pointwise mixing
        )

        self.bn_conv = nn.BatchNorm1d(self.n_conv_out)
        self.dropout_conv = nn.Dropout(dropout)

        # MLP head
        combined_dim = self.n_conv_out + n_tabular_features
        self.head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, url_chars, tabular):
        # Char embedding: (batch, max_len) → (batch, embed_dim, max_len)
        x = self.char_embed(url_chars).permute(0, 2, 1)

        # Parallel convolutions
        c3 = torch.relu(self.conv3(x))
        c5 = torch.relu(self.conv5(x))
        c7 = torch.relu(self.conv7(x))
        c11 = torch.relu(self.conv11(x))

        # Concatenate along filter dimension: (batch, n_filters*4, max_len)
        conv_out = torch.cat([c3, c5, c7, c11], dim=1)

        # Second conv block with residual connection
        conv_out = conv_out + self.conv_block2(conv_out)

        # Global max pool: (batch, n_filters*4)
        url_embedding = conv_out.max(dim=2).values

        url_embedding = self.bn_conv(url_embedding)
        url_embedding = self.dropout_conv(url_embedding)

        # Concatenate with tabular features
        combined = torch.cat([url_embedding, tabular], dim=1)

        return self.head(combined).squeeze(1)

    def extract_embedding(self, url_chars):
        """Extract the 1024-dim URL embedding without the MLP head."""
        with torch.no_grad():
            x = self.char_embed(url_chars).permute(0, 2, 1)
            c3 = torch.relu(self.conv3(x))
            c5 = torch.relu(self.conv5(x))
            c7 = torch.relu(self.conv7(x))
            c11 = torch.relu(self.conv11(x))
            conv_out = torch.cat([c3, c5, c7, c11], dim=1)
            conv_out = conv_out + self.conv_block2(conv_out)
            url_embedding = conv_out.max(dim=2).values
            return self.bn_conv(url_embedding)


# ============================================================
# Tweedie loss
# ============================================================

class TweedieLoss(nn.Module):
    """Tweedie deviance loss for zero-inflated right-skewed targets.

    Proper Tweedie loss with power parameter p ∈ (1, 2).
    Better than log-space MSE for this distribution.
    """

    def __init__(self, p=1.5):
        super().__init__()
        self.p = p

    def forward(self, y_pred, y_true):
        # Clamp predictions to be positive (Tweedie requires μ > 0)
        y_pred = torch.clamp(y_pred, min=1e-4)
        p = self.p
        # Tweedie deviance: 2 * (y^(2-p)/((1-p)(2-p)) - y*μ^(1-p)/(1-p) + μ^(2-p)/(2-p))
        # Simplified for gradient computation:
        loss = -y_true * torch.pow(y_pred, 1 - p) / (1 - p) + torch.pow(y_pred, 2 - p) / (2 - p)
        return loss.mean()


# ============================================================
# Feature engineering (minimal — CNN learns URL features)
# ============================================================

def engineer_tabular_features(df, train_df=None):
    """Build tabular features that complement the CNN.

    The CNN handles URL semantics. Tabular features provide:
    - Domain identity (target-encoded)
    - Resource type
    - Request metadata
    """
    features = pd.DataFrame(index=df.index)

    # Target-encode domain using train stats
    if train_df is not None:
        domain_medians = train_df.groupby('tracker_domain')['transfer_bytes'].median()
        domain_type_medians = train_df.groupby(['tracker_domain', 'resource_type'])['transfer_bytes'].median()
        global_median = train_df['transfer_bytes'].median()
    else:
        domain_medians = df.groupby('tracker_domain')['transfer_bytes'].median()
        domain_type_medians = df.groupby(['tracker_domain', 'resource_type'])['transfer_bytes'].median()
        global_median = df['transfer_bytes'].median()

    features['domain_median_bytes'] = df['tracker_domain'].map(domain_medians).fillna(global_median)
    features['domain_type_median'] = df.set_index(['tracker_domain', 'resource_type']).index.map(
        lambda x: domain_type_medians.get(x, domain_medians.get(x[0], global_median))
    )

    # URL structure (complements CNN — these are numeric summaries)
    features['path_depth'] = df['path_depth'].fillna(0)
    features['url_length'] = df['url_length'].fillna(0)
    features['num_query_params'] = df['num_query_params'].fillna(0)

    # Resource type one-hot
    for rt in ['script', 'image', 'other', 'html', 'text', 'css']:
        features[f'rt_{rt}'] = (df['resource_type'] == rt).astype(int)

    # File extension one-hot
    for ext in ['js', 'gif', 'png', 'jpg', 'html', 'php', 'json', 'css']:
        features[f'ext_{ext}'] = (df['file_extension'] == ext).astype(int)

    # Request metadata
    features['has_query_params'] = df['has_query_params'].astype(int)

    # Initiator type
    for it in ['script', 'parser', 'other']:
        features[f'init_{it}'] = (df['initiator_type'] == it).astype(int)

    return features


# ============================================================
# Training
# ============================================================

def train_epoch(model, loader, optimizer, loss_fn, device, grad_clip=1.0):
    model.train()
    total_loss = 0
    n = 0
    for url_chars, tabular, targets in loader:
        url_chars = url_chars.to(device)
        tabular = tabular.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(url_chars, tabular)
        loss = loss_fn(preds, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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


def extract_embeddings(model, loader, device):
    """Extract URL embeddings from trained model for all samples."""
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for url_chars, tabular, _ in loader:
            url_chars = url_chars.to(device)
            emb = model.extract_embedding(url_chars)
            all_embeddings.append(emb.cpu().numpy())
    return np.vstack(all_embeddings)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=str(ROOT / 'data' / 'raw' / 'per_request_full.csv'),
                        help='Path to CSV or directory of sharded CSVs')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--extract', action='store_true',
                        help='Extract embeddings from saved model instead of training')
    parser.add_argument('--model', type=str, default=str(MODELS / 'url_cnn_large_best.pt'))
    args = parser.parse_args()

    OUTPUT.mkdir(exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    data_path = Path(args.data)
    if data_path.is_dir():
        print(f"Loading sharded data from {data_path}")
        # For sharded data, load first shard to get column info
        first_shard = sorted(glob.glob(str(data_path / "*.csv*")))[0]
        sample_df = pd.read_csv(first_shard, nrows=100)
        tab_features = engineer_tabular_features(sample_df)
        feature_cols = list(tab_features.columns)
        # TODO: implement sharded training with ShardedTrackerDataset
        raise NotImplementedError("Sharded training not yet implemented — use single CSV for now")
    else:
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df):,} rows")

    # Clean targets
    df = df[df['transfer_bytes'].notna()].copy()
    df['transfer_bytes'] = df['transfer_bytes'].clip(lower=0)

    # Train/val/test split (row-level, stratified by domain)
    np.random.seed(42)
    domains = df['tracker_domain'].unique()
    np.random.shuffle(domains)
    n = len(domains)
    train_domains = set(domains[:int(0.7 * n)])
    val_domains = set(domains[int(0.7 * n):int(0.85 * n)])
    test_domains = set(domains[int(0.85 * n):])

    train_df = df[df['tracker_domain'].isin(train_domains)].reset_index(drop=True)
    val_df = df[df['tracker_domain'].isin(val_domains)].reset_index(drop=True)
    test_df = df[df['tracker_domain'].isin(test_domains)].reset_index(drop=True)
    print(f"Split: train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")

    # Feature engineering
    tab_train = engineer_tabular_features(train_df, train_df)
    tab_val = engineer_tabular_features(val_df, train_df)
    tab_test = engineer_tabular_features(test_df, train_df)
    feature_cols = list(tab_train.columns)
    n_features = len(feature_cols)
    print(f"Tabular features: {n_features}")

    # Datasets
    train_dataset = TrackerRequestDataset(
        train_df['url_path'].values, tab_train.values, train_df['transfer_bytes'].values)
    val_dataset = TrackerRequestDataset(
        val_df['url_path'].values, tab_val.values, val_df['transfer_bytes'].values)
    test_dataset = TrackerRequestDataset(
        test_df['url_path'].values, tab_test.values, test_df['transfer_bytes'].values)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    # Model
    model = URLCharCNNLarge(
        vocab_size=VOCAB_SIZE,
        embed_dim=64,
        n_tabular_features=n_features,
        n_filters=256,
        hidden_dim=512,
        dropout=0.3,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} ({trainable_params:,} trainable)")

    if args.extract:
        # Extract embeddings from saved model
        print(f"Loading model from {args.model}")
        model.load_state_dict(torch.load(args.model, weights_only=True, map_location=device))
        print("Extracting embeddings...")
        all_loader = DataLoader(
            TrackerRequestDataset(df['url_path'].values,
                                  engineer_tabular_features(df, train_df).values,
                                  df['transfer_bytes'].values),
            batch_size=args.batch_size, shuffle=False, num_workers=0)
        embeddings = extract_embeddings(model, all_loader, device)
        out_path = MODELS / 'url_embeddings.npy'
        np.save(out_path, embeddings)
        print(f"Saved {embeddings.shape} embeddings to {out_path}")
        return

    # Loss and optimizer
    loss_fn = TweedieLoss(p=1.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr * 10, epochs=args.epochs,
        steps_per_epoch=len(train_loader), pct_start=0.1)

    # Training loop
    best_mae = float('inf')
    best_epoch = 0
    patience = 7
    no_improve = 0

    print(f"\n{'Epoch':>5s}  {'Train Loss':>10s}  {'Val MAE':>10s}  {'Val rho':>10s}  {'Time':>6s}")
    print("-" * 50)

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_mae, val_rho, _, _ = evaluate_model(model, val_loader, device)
        elapsed = time.time() - t0

        marker = " *" if val_mae < best_mae else ""
        print(f"{epoch:5d}  {train_loss:10.4f}  {val_mae:10,.0f}  {val_rho:10.4f}  {elapsed:5.1f}s{marker}")

        if val_mae < best_mae:
            best_mae = val_mae
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), str(MODELS / 'url_cnn_large_best.pt'))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Final evaluation on test set
    model.load_state_dict(torch.load(str(MODELS / 'url_cnn_large_best.pt'), weights_only=True))
    test_mae, test_rho, test_preds, test_targets = evaluate_model(model, test_loader, device)

    print(f"\n{'='*60}")
    print(f"FINAL TEST RESULTS (best epoch: {best_epoch})")
    print(f"{'='*60}")
    print(f"URL CNN Large:  MAE={test_mae:>10,.0f}  rho={test_rho:.4f}")
    print(f"Parameters:     {total_params:,}")

    # Extract and save embeddings for entire dataset
    print("\nExtracting embeddings for full dataset...")
    full_tab = engineer_tabular_features(df, train_df)
    full_dataset = TrackerRequestDataset(
        df['url_path'].values, full_tab.values, df['transfer_bytes'].values)
    full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
    embeddings = extract_embeddings(model, full_loader, device)
    np.save(MODELS / 'url_embeddings.npy', embeddings)
    print(f"Saved {embeddings.shape} embeddings to {MODELS / 'url_embeddings.npy'}")

    # Save results
    results = {
        'test_mae': float(test_mae),
        'test_rho': float(test_rho),
        'best_epoch': best_epoch,
        'total_params': total_params,
        'n_train': len(train_df),
        'n_val': len(val_df),
        'n_test': len(test_df),
        'embed_dim': 64,
        'n_filters': 256,
        'hidden_dim': 512,
        'url_embedding_dim': 1024,
    }
    with open(MODELS / 'url_cnn_large_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {MODELS / 'url_cnn_large_results.json'}")


if __name__ == '__main__':
    main()
