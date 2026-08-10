"""Extract ordered sequences of ML function calls from Python files.

v2 improvements over v1:
  1. Repo blocklist — drop files from library-implementation repos (scikit-learn,
     pytorch, keras, tensorflow, xgboost, lightgbm). Their test suites use every
     estimator densely and were polluting neighborhoods with utility internals.
  2. Token stopword filter — drop names matching sklearn.utils.*, sklearn.tests.*,
     *estimator_checks*, sklearn.inspection._plot.*. These are library-internal
     scaffolding, not user-facing ML calls.
  3. Method-call resolution — track `var = SomeClass(...)` assignments and, when
     `var.method(...)` appears later, emit `SomeClass.method` as a token. This
     recovers the fit / predict / transform / compile / etc. that are the actual
     meat of ML pipelines and were invisible to the v1 extractor.

Each line of output: {"file": ..., "repo": ..., "tags": [...],
                      "seq": ["sklearn.cluster.KMeans", "sklearn.cluster.KMeans.fit", ...]}

Usage:
    python -m func2vec.extract --min-len 3
"""
from __future__ import annotations

import argparse
import ast
import json
import re
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
    "tf",
    "keras",
    "xgboost",
    "lightgbm",
    "numpy",
    "np",
    "pandas",
    "pd",
    "scipy",
)

COMMON_ALIAS_CANONICAL = {
    "np": "numpy",
    "pd": "pandas",
    "tf": "tensorflow",
    "plt": "matplotlib.pyplot",
}

# Repos to skip — their content is the library implementation itself, so
# test/example files use every estimator densely and skew the embedding toward
# library-internal scaffolding rather than end-user ML pipelines.
REPO_BLOCKLIST = {
    "scikit-learn/scikit-learn",
    "pytorch/pytorch",
    "keras-team/keras",
    "tensorflow/tensorflow",
    "dmlc/xgboost",
    "microsoft/lightgbm",
    "google/jax",
    "scipy/scipy",
    "numpy/numpy",
    "pandas-dev/pandas",
    "huggingface/transformers",  # its examples call every estimator too
}
REPO_BLOCKLIST = {r.lower() for r in REPO_BLOCKLIST}

# Token-name stopwords. Anything matching these regexes is dropped from the
# emitted sequence — internal scaffolding, not something a user would call.
TOKEN_STOPWORD_PATTERNS = [
    re.compile(r"^sklearn\.utils\."),
    re.compile(r"^sklearn\.tests?\."),
    re.compile(r"^sklearn\..*\._[a-z]"),  # sklearn.foo._private.*
    re.compile(r"\.estimator_checks\."),
    re.compile(r"^sklearn\.inspection\._plot\."),
    re.compile(r"^sklearn\._config\."),
    re.compile(r"^sklearn\.experimental\."),
    re.compile(r"^sklearn\.base\.is_"),  # is_classifier, is_regressor etc — used inside sklearn tests
    re.compile(r"^sklearn\.datasets\.samples_generator\."),
]

TASK_KEYWORDS = {
    "clustering": ("kmeans", "dbscan", "agglomerative", "spectral", "cluster", "meanshift", "optics", "affinity", "birch"),
    "classification": ("classifier", "logisticregression", "svc", "randomforestclassifier", "gradientboosting", "knneighbors", "xgbclassifier", "lgbmclassifier"),
    "regression": ("regressor", "linearregression", "ridge", "lasso", "elasticnet", "svr", "randomforestregressor", "xgbregressor", "lgbmregressor"),
    "neural_network": ("torch.nn", "keras.layers", "tf.keras", "conv2d", "dense", "lstm", "gru", "transformer", "sequential", "adam", "sgd"),
    "preprocessing": ("standardscaler", "minmaxscaler", "normalizer", "pca", "tsne", "onehotencoder", "labelencoder"),
    "evaluation": ("accuracy_score", "f1_score", "roc_auc_score", "mean_squared_error", "silhouette_score", "confusion_matrix"),
}


class ImportResolver(ast.NodeVisitor):
    def __init__(self) -> None:
        self.aliases: dict[str, str] = {}

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            local = alias.asname or alias.name.split(".")[0]
            self.aliases[local] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is None:
            return
        for alias in node.names:
            local = alias.asname or alias.name
            self.aliases[local] = f"{node.module}.{alias.name}"


def resolve_call_qualname(func: ast.AST, aliases: dict[str, str]) -> str | None:
    parts: list[str] = []
    cur = func
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    else:
        return None

    parts.reverse()
    base = parts[0]
    if base not in aliases:
        return ".".join(parts) if base in WHITELIST_PREFIXES else None
    resolved_base = aliases[base]
    resolved_base = COMMON_ALIAS_CANONICAL.get(resolved_base, resolved_base)
    return ".".join([resolved_base] + parts[1:])


def in_whitelist(qual: str) -> bool:
    top = qual.split(".")[0]
    return top in WHITELIST_PREFIXES


def is_stopword(qual: str) -> bool:
    return any(p.search(qual) for p in TOKEN_STOPWORD_PATTERNS)


def looks_like_class(qual: str) -> bool:
    """Heuristic: last segment starts with uppercase => class constructor."""
    last = qual.rsplit(".", 1)[-1]
    return bool(last) and last[0].isupper()


def _method_name(func: ast.AST) -> tuple[str, str] | None:
    """If func is `some_var.method` return (var_name, method_name)."""
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return func.value.id, func.attr
    return None


def extract_sequence(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    resolver = ImportResolver()
    resolver.visit(tree)
    aliases = resolver.aliases
    for k, v in COMMON_ALIAS_CANONICAL.items():
        aliases.setdefault(k, v)

    # var_name -> qualified class name assigned by `var = SomeClass(...)`
    var_class: dict[str, str] = {}

    seq: list[tuple[int, int, str]] = []

    for node in ast.walk(tree):
        # (a) Direct calls into whitelisted libraries.
        if isinstance(node, ast.Call):
            qual = resolve_call_qualname(node.func, aliases)
            if qual and in_whitelist(qual) and not is_stopword(qual):
                seq.append((node.lineno, node.col_offset, qual))

            # (b) Method calls on tracked instances -> emit ClassQual.method
            mn = _method_name(node.func)
            if mn is not None:
                var, method = mn
                if var in var_class:
                    combined = f"{var_class[var]}.{method}"
                    if not is_stopword(combined):
                        seq.append((node.lineno, node.col_offset, combined))

        # (c) Track `var = Class(...)` and `var = other_var` propagation.
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call_qual = resolve_call_qualname(node.value.func, aliases)
            if call_qual and in_whitelist(call_qual) and looks_like_class(call_qual) and not is_stopword(call_qual):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_class[target.id] = call_qual
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Name):
            src_name = node.value.id
            if src_name in var_class:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_class[target.id] = var_class[src_name]

    seq.sort()
    return [q for _, _, q in seq]


def tag_sequence(seq: list[str]) -> list[str]:
    joined = " ".join(seq).lower()
    return [tag for tag, kws in TASK_KEYWORDS.items() if any(kw in joined for kw in kws)]


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
    ap.add_argument("--min-len", type=int, default=3)
    args = ap.parse_args()

    manifest = load_manifest()
    files = sorted(RAW.glob("*"))
    kept = 0
    dropped_short = 0
    dropped_blocked = 0
    dropped_parse = 0
    with OUT.open("w") as out:
        for i, fp in enumerate(files):
            meta = manifest.get(fp.name, {})
            repo = (meta.get("repo") or "").lower()
            if repo in REPO_BLOCKLIST:
                dropped_blocked += 1
                continue
            try:
                src = fp.read_text(encoding="utf-8", errors="replace")
            except Exception:
                dropped_parse += 1
                continue
            seq = extract_sequence(src)
            if len(seq) < args.min_len:
                dropped_short += 1
                continue
            out.write(json.dumps({
                "file": fp.name,
                "repo": meta.get("repo"),
                "path": meta.get("path"),
                "tags": tag_sequence(seq),
                "seq": seq,
            }) + "\n")
            kept += 1
            if (i + 1) % 500 == 0:
                print(f"[extract] processed={i+1} kept={kept} short={dropped_short} blocked={dropped_blocked}", file=sys.stderr)
    print(f"[extract] done. kept={kept} short={dropped_short} blocked={dropped_blocked} parse-err={dropped_parse} -> {OUT}")


if __name__ == "__main__":
    main()
