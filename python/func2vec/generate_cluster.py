"""Generate a runnable clustering script driven by the func2vec embedding.

Given flags describing the clustering task, this script:

  1. Picks a clustering algorithm by heuristic on the flags (`--k`,
     `--dataset-size`, `--data-kind`, `--structure`).
  2. Uses the embedding's nearest-neighbor lookup, filtered to a small
     preprocessing whitelist and metrics whitelist, to pick companion
     estimators (StandardScaler / PCA / TfidfVectorizer) and evaluation
     metrics (silhouette_score, davies_bouldin_score, ...). If the
     embedding surfaces multiple, we take the ones that appear in vocab
     nearest to the picked algorithm.
  3. Reads the picked classes' real `__init__` signatures via `inspect` so
     the emitted parameter names actually exist in the installed library.
  4. Writes a single-file, runnable Python script to `--output`.

Deliberately not a black-box code model — this is a rules+embedding
synthesizer. The embedding contributes *which companions* to include based
on real GitHub co-occurrence data.

Usage:
    python -m func2vec.generate_cluster --k 5 --dataset-size large --output out.py
    python -m func2vec.generate_cluster --k unknown --data-kind text --output out.py
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from pathlib import Path

from gensim.models import Word2Vec

ROOT = Path(__file__).parent
MODELS = ROOT / "models"

# Small whitelists so the embedding picks *from* a known-safe menu. This
# keeps generated code sane even when the nearest-neighbor list drifts.
PREPROCESSING_WHITELIST = [
    "sklearn.preprocessing.StandardScaler",
    "sklearn.preprocessing.MinMaxScaler",
    "sklearn.preprocessing.RobustScaler",
    "sklearn.preprocessing.Normalizer",
    "sklearn.decomposition.PCA",
    "sklearn.decomposition.TruncatedSVD",
    "sklearn.feature_extraction.text.TfidfVectorizer",
    "sklearn.feature_extraction.text.CountVectorizer",
]

METRIC_WHITELIST = [
    "sklearn.metrics.silhouette_score",
    "sklearn.metrics.davies_bouldin_score",
    "sklearn.metrics.calinski_harabasz_score",
    "sklearn.metrics.adjusted_rand_score",
    "sklearn.metrics.normalized_mutual_info_score",
    "sklearn.metrics.homogeneity_score",
    "sklearn.metrics.completeness_score",
    "sklearn.metrics.v_measure_score",
]


def pick_algorithm(args) -> str:
    """Heuristic algorithm selection from flags."""
    if args.algorithm:
        return args.algorithm
    if args.structure == "hierarchical":
        return "sklearn.cluster.AgglomerativeClustering"
    if args.k == "unknown":
        return "sklearn.cluster.DBSCAN"
    if args.dataset_size == "large":
        return "sklearn.cluster.MiniBatchKMeans"
    if args.data_kind == "text":
        return "sklearn.cluster.KMeans"  # over TF-IDF vectors
    return "sklearn.cluster.KMeans"


def rank_companions(model: Word2Vec, seed: str, whitelist: list[str], k: int) -> list[str]:
    """Score each whitelist token by cosine to seed; return top-k in vocab."""
    if seed not in model.wv:
        return whitelist[:k]
    scores = []
    for tok in whitelist:
        if tok in model.wv:
            scores.append((float(model.wv.similarity(seed, tok)), tok))
    scores.sort(reverse=True)
    return [t for _, t in scores[:k]]


def import_class(qual: str):
    parts = qual.split(".")
    for split in range(len(parts) - 1, 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:split]))
        except Exception:
            continue
        try:
            for name in parts[split:]:
                obj = getattr(obj, name)
        except Exception:
            continue
        return obj
    return None


def safe_signature(cls, wanted: dict) -> dict:
    """Keep only wanted kwargs that the class's __init__ actually accepts."""
    try:
        sig = inspect.signature(cls)
    except (ValueError, TypeError):
        return {}
    params = set(sig.parameters.keys())
    return {k: v for k, v in wanted.items() if k in params}


def qual_split(qual: str) -> tuple[str, str]:
    """('sklearn.cluster.KMeans') → ('sklearn.cluster', 'KMeans')."""
    mod, name = qual.rsplit(".", 1)
    return mod, name


def render_kwargs(kw: dict) -> str:
    return ", ".join(f"{k}={render_value(v)}" for k, v in kw.items())


def render_value(v) -> str:
    if isinstance(v, str):
        return repr(v)
    return repr(v)


def build_script(args, model: Word2Vec) -> str:
    algo = pick_algorithm(args)
    algo_cls = import_class(algo)
    if algo_cls is None:
        raise SystemExit(f"could not import {algo}")
    algo_mod, algo_name = qual_split(algo)

    # Preprocessing choices — text corpora need a vectorizer even if scaling is off
    preprocs: list[tuple[str, dict]] = []
    if args.data_kind == "text":
        preprocs.append(("sklearn.feature_extraction.text.TfidfVectorizer",
                         safe_signature(import_class("sklearn.feature_extraction.text.TfidfVectorizer"),
                                        {"max_features": 5000, "stop_words": "english"})))
    if args.scale:
        # let embedding pick between Standard/MinMax/Robust based on nearness to algo
        picks = rank_companions(model, algo, [
            "sklearn.preprocessing.StandardScaler",
            "sklearn.preprocessing.MinMaxScaler",
            "sklearn.preprocessing.RobustScaler",
        ], k=1)
        if picks:
            cls = import_class(picks[0])
            preprocs.append((picks[0], safe_signature(cls, {})))
    if args.pca:
        preprocs.append(("sklearn.decomposition.PCA",
                         safe_signature(import_class("sklearn.decomposition.PCA"),
                                        {"n_components": args.pca, "random_state": 42})))

    # Algorithm kwargs
    wanted_algo_kwargs: dict = {"random_state": 42}
    if args.k not in (None, "unknown"):
        wanted_algo_kwargs["n_clusters"] = int(args.k)
    if algo.endswith("MiniBatchKMeans"):
        wanted_algo_kwargs["batch_size"] = 1024
    if algo.endswith("DBSCAN"):
        wanted_algo_kwargs["eps"] = 0.5
        wanted_algo_kwargs["min_samples"] = 5
    if algo.endswith("AgglomerativeClustering"):
        wanted_algo_kwargs["linkage"] = "ward"
    algo_kwargs = safe_signature(algo_cls, wanted_algo_kwargs)

    # Metrics — take top-2 by nearness to algo, or fallback to silhouette
    metrics = rank_companions(model, algo, METRIC_WHITELIST, k=3)
    if not metrics:
        metrics = ["sklearn.metrics.silhouette_score"]

    # Emit script
    imports: list[str] = ["import numpy as np"]
    if args.data_kind == "text":
        imports.append("from sklearn.datasets import fetch_20newsgroups")
    else:
        imports.append("from sklearn.datasets import make_blobs")
    for pre, _ in preprocs:
        mod, name = qual_split(pre)
        imports.append(f"from {mod} import {name}")
    imports.append(f"from {algo_mod} import {algo_name}")
    for m in metrics:
        mod, name = qual_split(m)
        imports.append(f"from {mod} import {name}")

    lines: list[str] = []
    lines.append('"""Generated by func2vec.generate_cluster.')
    lines.append(f"Intent: k={args.k} dataset_size={args.dataset_size} data_kind={args.data_kind} structure={args.structure}")
    lines.append(f"Picked algorithm: {algo}")
    lines.append(f"Preprocessing (embedding-picked where applicable): {[p for p, _ in preprocs] or 'none'}")
    lines.append(f"Evaluation (embedding-ranked): {metrics}")
    lines.append('"""')
    lines.append("")
    lines.extend(imports)
    lines.append("")
    lines.append("")
    lines.append("def load_data():")
    if args.data_kind == "text":
        lines.append("    ds = fetch_20newsgroups(subset='train', remove=('headers','footers','quotes'))")
        lines.append("    return ds.data, ds.target")
    else:
        n_samples = 20000 if args.dataset_size == "large" else 1500
        centers = int(args.k) if str(args.k).isdigit() else 4
        lines.append(f"    X, y = make_blobs(n_samples={n_samples}, centers={centers}, n_features=20,")
        lines.append("                      cluster_std=1.2, random_state=42)")
        lines.append("    return X, y")
    lines.append("")
    lines.append("")
    lines.append("def build_pipeline():")
    lines.append("    steps = []")
    for pre, kw in preprocs:
        _, name = qual_split(pre)
        lines.append(f"    steps.append({name}({render_kwargs(kw)}))")
    lines.append(f"    steps.append({algo_name}({render_kwargs(algo_kwargs)}))")
    lines.append("    return steps")
    lines.append("")
    lines.append("")
    lines.append("def run(X, y=None):")
    lines.append("    steps = build_pipeline()")
    lines.append("    Xt = X")
    lines.append("    for step in steps[:-1]:")
    lines.append("        Xt = step.fit_transform(Xt)")
    lines.append("    clusterer = steps[-1]")
    lines.append("    if hasattr(clusterer, 'fit_predict'):")
    lines.append("        labels = clusterer.fit_predict(Xt)")
    lines.append("    else:")
    lines.append("        clusterer.fit(Xt)")
    lines.append("        labels = clusterer.labels_")
    lines.append("")
    lines.append("    scoreable = labels[labels != -1] if -1 in np.unique(labels) else labels")
    lines.append("    Xs = Xt[labels != -1] if -1 in np.unique(labels) else Xt")
    lines.append("    n_clusters = len(set(scoreable))")
    lines.append("    print(f'algorithm    : {clusterer.__class__.__name__}')")
    lines.append("    print(f'n_clusters   : {n_clusters}')")
    lines.append("    if n_clusters >= 2:")
    for m in metrics:
        _, name = qual_split(m)
        if name in ("adjusted_rand_score", "normalized_mutual_info_score",
                    "homogeneity_score", "completeness_score", "v_measure_score"):
            lines.append(f"        if y is not None:")
            lines.append(f"            print(f'{name:<24}: {{{name}(y, labels):.4f}}')")
        else:
            lines.append(f"        print(f'{name:<24}: {{{name}(Xs, scoreable):.4f}}')")
    lines.append("    else:")
    lines.append("        print('too few clusters formed to score')")
    lines.append("    return labels")
    lines.append("")
    lines.append("")
    lines.append("if __name__ == '__main__':")
    if args.data_kind == "text":
        lines.append("    X, y = load_data()")
        lines.append("    # Text pipeline: the vectorizer converts strings to a matrix.")
        lines.append("    run(X, y)")
    else:
        lines.append("    X, y = load_data()")
        lines.append("    run(X, y)")

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", default=5, help="number of clusters, or 'unknown' for density-based")
    ap.add_argument("--data-kind", default="tabular", choices=["tabular", "text", "image"])
    ap.add_argument("--dataset-size", default="small", choices=["small", "large"])
    ap.add_argument("--structure", default="partition", choices=["partition", "hierarchical", "density"])
    ap.add_argument("--scale", action="store_true", help="include a scaler in the pipeline")
    ap.add_argument("--pca", type=int, default=0, help="if >0, include PCA with that many components")
    ap.add_argument("--algorithm", help="force a specific algorithm token (e.g. sklearn.cluster.SpectralClustering)")
    ap.add_argument("--model", default="func2vec_v4", help="which trained model under models/ to load")
    ap.add_argument("--output", required=True, help="write the generated .py here")
    args = ap.parse_args()

    model_path = MODELS / f"{args.model}.model"
    if not model_path.exists():
        # fallback to whatever's present
        for name in ("func2vec_v4", "func2vec_v3", "func2vec"):
            p = MODELS / f"{name}.model"
            if p.exists():
                print(f"[generate] {model_path.name} not found, using {p.name}", file=sys.stderr)
                model_path = p
                break
    model = Word2Vec.load(str(model_path))

    src = build_script(args, model)
    Path(args.output).write_text(src)
    print(f"[generate] wrote {args.output} using model={model_path.name}", file=sys.stderr)


if __name__ == "__main__":
    main()
