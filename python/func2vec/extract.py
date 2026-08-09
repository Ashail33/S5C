"""Extract ordered sequences of ML function calls from Python files.

Reads every file in data/raw/, resolves import aliases, walks the AST in source order,
emits qualified names of calls into whitelisted ML libraries. Writes data/sequences.jsonl.
Each line: {"file": ..., "repo": ..., "tags": [...], "seq": ["sklearn.cluster.KMeans", ...]}

Usage:
    python -m func2vec.extract
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RAW = DATA / "raw"
MANIFEST = DATA / "manifest.jsonl"
OUT = DATA / "sequences.jsonl"

WHITELIST_PREFIXES = (
    "sklearn",
    "torch",
    "tensorflow",
    "tf",  # common alias, resolved to tensorflow below
    "keras",
    "xgboost",
    "lightgbm",
    "numpy",
    "np",
    "pandas",
    "pd",
    "scipy",
)

# alias -> canonical top-level module for whitelisting/pretty-printing
COMMON_ALIAS_CANONICAL = {
    "np": "numpy",
    "pd": "pandas",
    "tf": "tensorflow",
    "plt": "matplotlib.pyplot",
}

TASK_KEYWORDS = {
    "clustering": ("kmeans", "dbscan", "agglomerative", "spectral", "cluster", "meanshift", "optics", "affinity", "birch"),
    "classification": ("classifier", "logisticregression", "svc", "randomforestclassifier", "gradientboosting", "knneighbors", "xgbclassifier", "lgbmclassifier"),
    "regression": ("regressor", "linearregression", "ridge", "lasso", "elasticnet", "svr", "randomforestregressor", "xgbregressor", "lgbmregressor"),
    "neural_network": ("torch.nn", "keras.layers", "tf.keras", "conv2d", "dense", "lstm", "gru", "transformer", "sequential", "adam", "sgd"),
    "preprocessing": ("standardscaler", "minmaxscaler", "normalizer", "pca", "tsne", "onehotencoder", "labelencoder"),
    "evaluation": ("accuracy_score", "f1_score", "roc_auc_score", "mean_squared_error", "silhouette_score", "confusion_matrix"),
}


class ImportResolver(ast.NodeVisitor):
    """Build a map: local name -> fully qualified module or object."""

    def __init__(self) -> None:
        # name -> qualified string (e.g. "np" -> "numpy", "KMeans" -> "sklearn.cluster.KMeans")
        self.aliases: dict[str, str] = {}

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            local = alias.asname or alias.name.split(".")[0]
            self.aliases[local] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is None:  # relative import, skip
            return
        for alias in node.names:
            local = alias.asname or alias.name
            self.aliases[local] = f"{node.module}.{alias.name}"


def resolve_call_qualname(func: ast.AST, aliases: dict[str, str]) -> str | None:
    """Turn a Call.func node into a qualified string using known aliases."""
    parts: list[str] = []
    cur = func
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    else:
        return None  # too dynamic to resolve

    parts.reverse()  # base ... .attr .attr
    base = parts[0]
    if base not in aliases:
        # Uncommon: base is not imported; fall back to raw base
        return ".".join(parts) if base in WHITELIST_PREFIXES else None
    resolved_base = aliases[base]
    # normalize common aliases even if user wrote them literally
    resolved_base = COMMON_ALIAS_CANONICAL.get(resolved_base, resolved_base)
    qual = ".".join([resolved_base] + parts[1:])
    return qual


def in_whitelist(qual: str) -> bool:
    top = qual.split(".")[0]
    return top in WHITELIST_PREFIXES


def extract_sequence(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    resolver = ImportResolver()
    resolver.visit(tree)
    aliases = resolver.aliases
    # add canonical aliases so unaliased `pd.read_csv` works even without `import pandas as pd`
    for k, v in COMMON_ALIAS_CANONICAL.items():
        aliases.setdefault(k, v)

    seq: list[tuple[int, int, str]] = []  # (line, col, qual)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            qual = resolve_call_qualname(node.func, aliases)
            if qual and in_whitelist(qual):
                seq.append((node.lineno, node.col_offset, qual))
    seq.sort()
    return [q for _, _, q in seq]


def tag_sequence(seq: list[str]) -> list[str]:
    joined = " ".join(seq).lower()
    tags = []
    for tag, kws in TASK_KEYWORDS.items():
        if any(kw in joined for kw in kws):
            tags.append(tag)
    return tags


def load_manifest() -> dict[str, dict]:
    m: dict[str, dict] = {}
    if not MANIFEST.exists():
        return m
    with MANIFEST.open() as f:
        for line in f:
            try:
                row = json.loads(line)
                m[row["local"]] = row
            except Exception:
                continue
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-len", type=int, default=3, help="drop files with fewer than N whitelisted calls")
    args = ap.parse_args()

    manifest = load_manifest()
    files = sorted(RAW.glob("*"))
    kept = 0
    dropped = 0
    with OUT.open("w") as out:
        for i, fp in enumerate(files):
            try:
                src = fp.read_text(encoding="utf-8", errors="replace")
            except Exception:
                dropped += 1
                continue
            seq = extract_sequence(src)
            if len(seq) < args.min_len:
                dropped += 1
                continue
            meta = manifest.get(fp.name, {})
            out.write(json.dumps({
                "file": fp.name,
                "repo": meta.get("repo"),
                "path": meta.get("path"),
                "tags": tag_sequence(seq),
                "seq": seq,
            }) + "\n")
            kept += 1
            if (i + 1) % 500 == 0:
                print(f"[extract] processed={i+1} kept={kept} dropped={dropped}", file=sys.stderr)
    print(f"[extract] done. kept={kept} dropped={dropped} -> {OUT}")


if __name__ == "__main__":
    main()
