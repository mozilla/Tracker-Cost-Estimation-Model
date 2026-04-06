"""
URL embedding via TF-IDF + Truncated SVD.

Tokenizes URL paths into segments, computes TF-IDF vectors,
then reduces to dense embeddings with SVD. Produces a fixed-size
vector per URL that captures semantic content without hand-crafted
regex features.

Usage:
  # Fit on training data, transform all data:
  from url_embeddings import URLEmbedder
  embedder = URLEmbedder(n_components=50)
  embedder.fit(train_urls)
  X_embed = embedder.transform(all_urls)

  # Or from CLI:
  python src/model/url_embeddings.py --data data/raw/per_request_full.csv
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from pathlib import Path
import joblib
import time

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models" / "per_request"


def tokenize_url(url):
    """Split a URL path into meaningful tokens.

    /gtag/js?id=UA-12345&v=3  →  "gtag js id UA 12345 v 3"

    Splits on: / ? & = . - _ and digits-vs-letters boundaries.
    Lowercases everything. Drops tokens shorter than 2 chars.
    """
    if pd.isna(url) or not isinstance(url, str):
        return ""
    # Split on URL delimiters
    tokens = re.split(r'[/\?&=\.\-_]+', url)
    # Further split camelCase and digit boundaries
    expanded = []
    for token in tokens:
        parts = re.sub(r'([a-z])([A-Z])', r'\1 \2', token)
        parts = re.sub(r'([A-Za-z])(\d)', r'\1 \2', parts)
        parts = re.sub(r'(\d)([A-Za-z])', r'\1 \2', parts)
        expanded.extend(parts.lower().split())
    # Drop short tokens (single chars are noise — version numbers, etc.)
    return " ".join(t for t in expanded if len(t) >= 2)


class URLEmbedder:
    """TF-IDF + SVD URL embedding pipeline.

    Transforms raw URL path strings into dense n_components-dim vectors.
    """

    def __init__(self, n_components=50, max_features=50000,
                 ngram_range=(1, 2), min_df=5):
        """
        Args:
            n_components: Output embedding dimensionality.
            max_features: Max vocabulary size for TF-IDF.
            ngram_range: Token n-gram range. (1,2) captures both
                         individual tokens and bigrams like "gtag js".
            min_df: Minimum document frequency to include a token.
        """
        self.n_components = n_components
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                analyzer='word',
                preprocessor=tokenize_url,
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                sublinear_tf=True,  # use log(1 + tf) instead of raw tf
                dtype=np.float32,
            )),
            ('svd', TruncatedSVD(
                n_components=n_components,
                random_state=42,
            )),
        ])
        self.is_fitted = False

    def fit(self, urls):
        """Fit TF-IDF vocabulary and SVD components on training URLs."""
        self.pipeline.fit(urls)
        self.is_fitted = True
        # Store explained variance for diagnostics
        self.explained_variance_ratio_ = self.pipeline['svd'].explained_variance_ratio_
        self.total_variance_explained_ = self.explained_variance_ratio_.sum()
        return self

    def transform(self, urls):
        """Transform URLs to dense embeddings."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first")
        return self.pipeline.transform(urls).astype(np.float32)

    def fit_transform(self, urls):
        """Fit and transform in one step."""
        self.is_fitted = True
        result = self.pipeline.fit_transform(urls).astype(np.float32)
        self.explained_variance_ratio_ = self.pipeline['svd'].explained_variance_ratio_
        self.total_variance_explained_ = self.explained_variance_ratio_.sum()
        return result

    def save(self, path=None):
        """Save fitted pipeline to disk."""
        path = path or MODELS / 'url_embedder.joblib'
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        print(f"Saved URL embedder to {path}")

    def load(self, path=None):
        """Load fitted pipeline from disk."""
        path = path or MODELS / 'url_embedder.joblib'
        self.pipeline = joblib.load(path)
        self.is_fitted = True
        self.explained_variance_ratio_ = self.pipeline['svd'].explained_variance_ratio_
        self.total_variance_explained_ = self.explained_variance_ratio_.sum()
        return self

    def get_top_terms(self, component=0, n=20):
        """Show top terms for a given SVD component (for interpretability)."""
        tfidf = self.pipeline['tfidf']
        svd = self.pipeline['svd']
        terms = tfidf.get_feature_names_out()
        weights = svd.components_[component]
        top_pos = np.argsort(weights)[-n:][::-1]
        top_neg = np.argsort(weights)[:n]
        print(f"Component {component} (explains {self.explained_variance_ratio_[component]*100:.1f}% variance)")
        print(f"  Positive: {', '.join(f'{terms[i]} ({weights[i]:.3f})' for i in top_pos)}")
        print(f"  Negative: {', '.join(f'{terms[i]} ({weights[i]:.3f})' for i in top_neg)}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=str(ROOT / 'data' / 'raw' / 'per_request_full.csv'))
    parser.add_argument('--n-components', type=int, default=50)
    parser.add_argument('--url-col', type=str, default='url_path')
    args = parser.parse_args()

    MODELS.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.data}")
    df = pd.read_csv(args.data, usecols=[args.url_col])
    urls = df[args.url_col].fillna("")
    print(f"Loaded {len(urls):,} URLs ({df[args.url_col].isna().sum()} null)")

    # Show tokenization examples
    print("\nTokenization examples:")
    samples = urls.dropna().sample(5, random_state=42)
    for url in samples:
        print(f"  {url[:80]}")
        print(f"    → {tokenize_url(url)}")

    # Fit embedder
    embedder = URLEmbedder(n_components=args.n_components)
    print(f"\nFitting TF-IDF + SVD (n_components={args.n_components})...")
    t0 = time.time()
    embeddings = embedder.fit_transform(urls)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Total variance explained: {embedder.total_variance_explained_*100:.1f}%")
    print(f"Top 5 components: {embedder.explained_variance_ratio_[:5]*100}")

    # Show what the components capture
    print("\nTop SVD components:")
    for i in range(min(5, args.n_components)):
        embedder.get_top_terms(component=i, n=10)
        print()

    # Save
    embedder.save()
    np.save(MODELS / 'url_embeddings_tfidf.npy', embeddings)
    print(f"Saved embeddings to {MODELS / 'url_embeddings_tfidf.npy'}")

    # Vocab stats
    tfidf = embedder.pipeline['tfidf']
    vocab_size = len(tfidf.vocabulary_)
    print(f"\nVocabulary size: {vocab_size:,}")


if __name__ == '__main__':
    main()
