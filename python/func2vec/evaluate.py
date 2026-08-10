"""Sanity checks and demos over a trained func2vec model.

Prints:
- Nearest neighbors for a handful of anchor functions across ML task families.
- Silhouette score of the embedding partitioned by task-tag (higher = task families separate).
- Optional t-SNE plot to models/tsne.png if matplotlib is available.

Usage:
    python -m func2vec.evaluate
    python -m func2vec.evaluate --anchors sklearn.cluster.KMeans torch.nn.Conv2d
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
DATA = ROOT / "data"

DEFAULT_ANCHORS = [
    "sklearn.cluster.KMeans",
    "sklearn.cluster.DBSCAN",
    "sklearn.linear_model.LogisticRegression",
    "sklearn.linear_model.LinearRegression",
    "sklearn.svm.SVC",
    "sklearn.ensemble.RandomForestClassifier",
    "sklearn.decomposition.PCA",
    "torch.nn.Conv2d",
    "torch.nn.Linear",
    "keras.layers.Dense",
    "xgboost.XGBClassifier",
]

TASK_KEYWORDS = {
    "clustering": ("kmeans", "dbscan", "agglomerative", "spectral", "meanshift", "optics", "birch"),
    "classification": ("classifier", "logisticregression", "svc", "xgbclassifier", "lgbmclassifier"),
    "regression": ("regressor", "linearregression", "ridge", "lasso", "elasticnet", "svr"),
    "neural_network": ("torch.nn.", "keras.layers.", "tf.keras.", "conv2d", "lstm", "dense"),
    "preprocessing": ("standardscaler", "minmaxscaler", "normalizer", ".pca", "tsne", "onehotencoder"),
    "evaluation": ("accuracy_score", "f1_score", "roc_auc_score", "mean_squared_error", "silhouette_score"),
}


def token_task(token: str) -> str | None:
    low = token.lower()
    for tag, kws in TASK_KEYWORDS.items():
        if any(kw in low for kw in kws):
            return tag
    return None


def neighbors(model: Word2Vec, anchors: list[str], topn: int = 8) -> None:
    print("\n=== nearest neighbors ===")
    for a in anchors:
        if a not in model.wv:
            print(f"[skip] {a} not in vocab")
            continue
        nbrs = model.wv.most_similar(a, topn=topn)
        print(f"\n{a}")
        for tok, score in nbrs:
            print(f"  {score:0.3f}  {tok}")


def task_silhouette(model: Word2Vec) -> None:
    print("\n=== task-tag silhouette ===")
    X, labels = [], []
    for tok in model.wv.index_to_key:
        t = token_task(tok)
        if t is None:
            continue
        X.append(model.wv[tok])
        labels.append(t)
    if len({*labels}) < 2 or len(X) < 20:
        print(f"[silhouette] insufficient tagged tokens ({len(X)}, {len(set(labels))} tags)")
        return
    X = np.stack(X)
    s = silhouette_score(X, labels, metric="cosine")
    counts = defaultdict(int)
    for l in labels:
        counts[l] += 1
    print(f"silhouette={s:0.4f}  n={len(X)}  by-tag={dict(counts)}")


def try_tsne(model: Word2Vec) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except Exception as e:
        print(f"[tsne] skipping: {e}")
        return
    X, labels, words = [], [], []
    for tok in model.wv.index_to_key:
        t = token_task(tok)
        if t is None:
            continue
        X.append(model.wv[tok])
        labels.append(t)
        words.append(tok)
    if len(X) < 10:
        print("[tsne] too few tagged tokens")
        return
    X = np.stack(X)
    perp = min(30, max(5, len(X) // 4))
    coords = TSNE(n_components=2, perplexity=perp, init="pca", random_state=0).fit_transform(X)
    fig, ax = plt.subplots(figsize=(10, 8))
    tags = sorted(set(labels))
    for tag in tags:
        idx = [i for i, l in enumerate(labels) if l == tag]
        ax.scatter(coords[idx, 0], coords[idx, 1], label=tag, s=25, alpha=0.75)
    ax.legend()
    ax.set_title("func2vec — t-SNE of tagged tokens")
    out = MODELS / "tsne.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[tsne] wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchors", nargs="*", default=None)
    ap.add_argument("--topn", type=int, default=8)
    ap.add_argument("--no-tsne", action="store_true")
    ap.add_argument("--model", default="func2vec", help="which model file under models/ to load (default: func2vec)")
    args = ap.parse_args()

    model = Word2Vec.load(str(MODELS / f"{args.model}.model"))
    anchors = args.anchors or DEFAULT_ANCHORS
    neighbors(model, anchors, topn=args.topn)
    task_silhouette(model)
    if not args.no_tsne:
        try_tsne(model)


if __name__ == "__main__":
    main()
