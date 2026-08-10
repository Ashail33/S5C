"""Decompose each ML algorithm class into its internal-call components.

For every class token in the current trained vocabulary that lives in an
installed library, we:
  1. Import the class object via importlib.
  2. Grab its source with inspect.getsource.
  3. Walk each method definition's AST body and collect every function/method
     the method calls internally (including private helpers like _kmeans_plusplus).
  4. Also walk the module the class lives in, so we can resolve those helper
     names to their qualified module paths where possible.

Output: data/decomposition.jsonl, one line per class:
  {
    "class": "sklearn.cluster.KMeans",
    "n_methods": 12,
    "components": ["_kmeans_plusplus", "_labels_inertia", ...],   # top by frequency
    "method_components": {"fit": [...], "predict": [...], ...}
  }

Usage:
    python -m func2vec.decompose --top 20
"""
from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUT = DATA / "decomposition.jsonl"

# Libraries we're willing to import for decomposition. Kept small to control
# what actually loads — heavier stacks (torch, tensorflow, keras) can be added
# in an env where they're installed.
SUPPORTED_TOP_MODULES = {
    "sklearn",
    "xgboost",
    "lightgbm",
    "scipy",
    "numpy",
    "pandas",
}

# Skip these method names when collecting components — they're too generic
# (constructors, dunder plumbing) to be discriminative.
SKIP_METHODS = {"__init__", "__new__", "__repr__", "__str__", "__eq__", "__hash__", "__len__", "__iter__", "__getitem__", "__setitem__", "__contains__", "__enter__", "__exit__"}


def load_vocab_classes() -> list[str]:
    """Return classes from the trained model's metadata.tsv."""
    md = MODELS / "metadata.tsv"
    if not md.exists():
        raise SystemExit(f"missing {md} — run train.py first")
    classes: list[str] = []
    with md.open() as f:
        next(f)  # header
        for line in f:
            word = line.split("\t", 1)[0]
            # class heuristic: last segment starts uppercase, and the token has
            # no `.method` suffix (a class, not a bound method)
            parts = word.split(".")
            last = parts[-1]
            if not last or not last[0].isupper():
                continue
            # skip method tokens we synthesized in v2 — those have Uppercase.method
            # but we can detect them by having an Uppercase segment NOT at the end.
            has_uppercase_middle = any(p and p[0].isupper() for p in parts[:-1])
            if has_uppercase_middle:
                continue
            if parts[0] not in SUPPORTED_TOP_MODULES:
                continue
            classes.append(word)
    return classes


def import_class(qual: str):
    """Turn 'sklearn.cluster.KMeans' into the class object, or None."""
    parts = qual.split(".")
    for split in range(len(parts) - 1, 0, -1):
        mod_name = ".".join(parts[:split])
        attr_path = parts[split:]
        try:
            obj = importlib.import_module(mod_name)
        except Exception:
            continue
        try:
            for name in attr_path:
                obj = getattr(obj, name)
        except (AttributeError, ImportError, Exception):
            continue
        if inspect.isclass(obj):
            return obj
        # If it's a re-export of a class, follow one more step
        if inspect.ismodule(obj):
            continue
    return None


def collect_method_calls(method_source: str) -> list[str]:
    """Parse a method body and return call names (attr names or bare names)."""
    try:
        tree = ast.parse(inspect.cleandoc(method_source))
    except SyntaxError:
        return []
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute):
                names.append(f.attr)
            elif isinstance(f, ast.Name):
                names.append(f.id)
    return names


def decompose_class(cls) -> dict[str, list[str]]:
    """Return {method_name: [called_name, ...]} for this class."""
    out: dict[str, list[str]] = {}
    for name, meth in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name in SKIP_METHODS:
            continue
        # Skip methods inherited from `object` or from other base classes we
        # don't want (dunder plumbing dominates otherwise).
        try:
            defining_cls = getattr(inspect.unwrap(meth), "__qualname__", "").split(".")[0]
        except Exception:
            defining_cls = ""
        # keep methods whose __qualname__ starts with this class name OR its bases
        # — good enough heuristic to filter out `object.__reduce__` etc.
        try:
            src = inspect.getsource(meth)
        except (OSError, TypeError):
            continue
        calls = collect_method_calls(src)
        if calls:
            out[name] = calls
    return out


# Bare method names — components should be the helpers a method calls, not the
# method itself. When a class calls `.fit()` or `.predict()` inside another
# method, those are already represented in v2 by the Class.method tokens.
INHERITED_METHODS = {"fit", "predict", "transform", "fit_transform", "fit_predict",
                     "score", "decision_function", "predict_proba", "predict_log_proba",
                     "get_params", "set_params", "partial_fit", "inverse_transform"}

NOISE = {"len", "range", "print", "isinstance", "hasattr", "getattr", "setattr",
         "super", "min", "max", "sum", "int", "float", "str", "list", "dict",
         "tuple", "set", "type", "iter", "next", "map", "filter", "zip",
         "enumerate", "sorted", "reversed", "callable", "bool", "abs", "round",
         "any", "all", "id", "vars", "repr", "hash", "copy", "deepcopy",
         "format", "join", "append", "extend", "insert", "pop", "get", "keys",
         "values", "items", "update", "clear", "add", "remove", "discard",
         "setdefault", "startswith", "endswith", "split", "strip", "lower",
         "upper", "replace", "count", "index", "sort", "reverse", "warn",
         "warns", "warning", "warnings"}
# Python built-in exceptions — never a meaningful "component"
EXCEPTIONS = {"ValueError", "TypeError", "KeyError", "IndexError", "AttributeError",
              "RuntimeError", "NotImplementedError", "StopIteration", "OSError",
              "ImportError", "IOError", "FileNotFoundError", "ArithmeticError",
              "AssertionError", "LookupError", "NameError", "ZeroDivisionError"}


def summarize_raw(method_components: dict[str, list[str]]) -> Counter:
    """Per-class Counter of internal call names, minus noise."""
    counts: Counter = Counter()
    for calls in method_components.values():
        for name in calls:
            if name in NOISE or name in EXCEPTIONS or name in INHERITED_METHODS:
                continue
            if name.startswith("__"):
                continue
            counts[name] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=15, help="top-N distinctive components per class")
    ap.add_argument("--max-df", type=float, default=0.30, help="drop components appearing in more than this fraction of classes (scaffolding)")
    ap.add_argument("--min-df", type=int, default=1, help="drop components appearing in fewer than this many classes (singletons if <1)")
    args = ap.parse_args()

    classes = load_vocab_classes()
    print(f"[decompose] {len(classes)} class tokens from installed libraries", file=sys.stderr)

    # Pass 1: resolve every class and collect raw per-class component Counters.
    raw: dict[str, tuple[int, dict[str, list[str]], Counter]] = {}
    for qual in classes:
        cls = import_class(qual)
        if cls is None:
            continue
        method_components = decompose_class(cls)
        if not method_components:
            continue
        raw[qual] = (len(method_components), method_components, summarize_raw(method_components))
    print(f"[decompose] resolved {len(raw)} classes", file=sys.stderr)

    # Pass 2: document frequency across classes -> keep distinctive components.
    df: Counter = Counter()
    for _, _, counts in raw.values():
        for name in counts:
            df[name] += 1
    n_docs = max(1, len(raw))
    max_df_abs = int(args.max_df * n_docs)
    keep = {name for name, d in df.items() if args.min_df <= d <= max_df_abs}
    print(f"[decompose] after df filter (min={args.min_df}, max={max_df_abs}/{n_docs}): {len(keep)}/{len(df)} components retained", file=sys.stderr)

    # Pass 3: write distinctive top-K per class.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with OUT.open("w") as f:
        for qual, (n_meth, method_components, counts) in raw.items():
            distinctive = [(n, c) for n, c in counts.most_common() if n in keep]
            top = [n for n, _ in distinctive[: args.top]]
            if not top:
                continue
            f.write(json.dumps({
                "class": qual,
                "n_methods": n_meth,
                "components": top,
                "method_components": {m: cs[: args.top] for m, cs in method_components.items()},
            }) + "\n")
            written += 1
    print(f"[decompose] done. classes={len(classes)} resolved={len(raw)} written={written} -> {OUT}")


if __name__ == "__main__":
    main()
