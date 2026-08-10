"""Math-level decomposition — pull out only the numpy / scipy / math calls
each ML class actually makes, so the resulting substrate is the *math it does*
rather than the *framework scaffolding around it*.

How it resolves callables cleanly (better than raw AST name harvesting):
  1. For a class like `sklearn.cluster.KMeans`, get the module it lives in
     (`sklearn.cluster._kmeans`).
  2. For each call `foo(...)` in a method body, look up `foo` in the module's
     `__dict__` (bare Name calls) or `mod.foo.bar` (Attribute calls). This gives
     us the actual callable object, imported at whatever alias.
  3. Read `callable.__module__` — that's the true source library.
  4. If the source library starts with numpy / scipy / math, emit
     MATH:<name>. That's the "math substrate" this algorithm relies on.

Output: data/math_decomposition.jsonl
  { "class": "sklearn.cluster.KMeans",
    "math": ["MATH:mean", "MATH:sqrt", "MATH:argmin", ...],
    "math_by_module": {"numpy": [...], "scipy.sparse": [...]} }

Usage:
    python -m func2vec.math_decompose --top 15
"""
from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUT = DATA / "math_decomposition.jsonl"

MATH_ROOT_MODULES = ("numpy", "scipy", "math", "statistics", "cmath")

SUPPORTED_TOP_MODULES = {
    "sklearn", "xgboost", "lightgbm", "scipy", "numpy", "pandas",
}

SKIP_METHODS = {"__init__", "__new__", "__repr__", "__str__", "__eq__",
                "__hash__", "__len__", "__iter__", "__getitem__", "__setitem__",
                "__contains__", "__enter__", "__exit__"}


def load_vocab_classes() -> list[str]:
    md = MODELS / "metadata.tsv"
    if not md.exists():
        raise SystemExit(f"missing {md} — run train.py first")
    out: list[str] = []
    with md.open() as f:
        next(f)
        for line in f:
            word = line.split("\t", 1)[0]
            parts = word.split(".")
            last = parts[-1]
            if not last or not last[0].isupper():
                continue
            if any(p and p[0].isupper() for p in parts[:-1]):
                continue
            if parts[0] not in SUPPORTED_TOP_MODULES:
                continue
            out.append(word)
    return out


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
        if inspect.isclass(obj):
            return obj
    return None


def resolve_call(func: ast.AST, mod_dict: dict):
    """Return the actual callable object, or None if we can't resolve it."""
    if isinstance(func, ast.Name):
        return mod_dict.get(func.id)
    if isinstance(func, ast.Attribute):
        # Walk chain: base.attr1.attr2 ...
        parts: list[str] = []
        cur = func
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if not isinstance(cur, ast.Name):
            return None
        parts.reverse()
        obj = mod_dict.get(cur.id)
        if obj is None:
            return None
        for name in parts:
            try:
                obj = getattr(obj, name)
            except Exception:
                return None
        return obj
    return None


def is_math_callable(cb) -> tuple[bool, str, str]:
    """Return (is_math, root_module, short_name)."""
    if cb is None:
        return False, "", ""
    mod = getattr(cb, "__module__", "") or ""
    if not mod:
        # Sometimes numpy ufuncs have no __module__; fall back to type.
        mod = type(cb).__module__ or ""
    root = mod.split(".", 1)[0]
    if root not in MATH_ROOT_MODULES:
        return False, "", ""
    name = getattr(cb, "__name__", None) or getattr(cb, "__qualname__", "")
    if not name:
        return False, "", ""
    return True, mod, name


def _collect_from_source(src: str, mod_dict: dict, total: Counter, per_module: defaultdict):
    try:
        tree = ast.parse(inspect.cleandoc(src))
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        cb = resolve_call(node.func, mod_dict)
        is_math, mod, sname = is_math_callable(cb)
        if not is_math:
            continue
        total[sname] += 1
        key = mod.split(".", 1)[0]
        if "." in mod:
            key = key + "." + mod.split(".", 2)[1]
        per_module[key][sname] += 1


def decompose_math(cls) -> tuple[Counter, dict[str, list[str]]]:
    """Extract math substrate by walking:
      (1) all methods of the class itself, and
      (2) every module-level function in the class's defining module —
          this catches Cython/private helpers (_kmeans_plusplus, dbscan_inner, ...)
          that carry most of the actual math a modern sklearn class relies on.
    """
    try:
        defining_mod = importlib.import_module(cls.__module__)
    except Exception:
        return Counter(), {}
    mod_dict = getattr(defining_mod, "__dict__", {})

    total: Counter = Counter()
    per_module: defaultdict[str, Counter] = defaultdict(Counter)

    # (1) methods of the class
    for name, meth in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name in SKIP_METHODS:
            continue
        try:
            src = inspect.getsource(meth)
        except (OSError, TypeError):
            continue
        _collect_from_source(src, mod_dict, total, per_module)

    # (2) module-level helpers defined in the same file — these hold the
    # heavy Python-level math for most sklearn / scipy estimators.
    try:
        src_file = inspect.getsourcefile(cls)
    except (OSError, TypeError):
        src_file = None
    for name, obj in mod_dict.items():
        if not inspect.isfunction(obj):
            continue
        try:
            obj_file = inspect.getsourcefile(obj)
        except (OSError, TypeError):
            continue
        if src_file is not None and obj_file != src_file:
            continue
        try:
            src = inspect.getsource(obj)
        except (OSError, TypeError):
            continue
        _collect_from_source(src, mod_dict, total, per_module)

    by_module_top = {m: [n for n, _ in c.most_common(10)] for m, c in per_module.items()}
    return total, by_module_top


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    classes = load_vocab_classes()
    print(f"[math_decompose] {len(classes)} class tokens to introspect", file=sys.stderr)

    written = 0
    resolved = 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for qual in classes:
            cls = import_class(qual)
            if cls is None:
                continue
            resolved += 1
            counts, by_module = decompose_math(cls)
            if not counts:
                continue
            top = [f"MATH:{n}" for n, _ in counts.most_common(args.top)]
            f.write(json.dumps({
                "class": qual,
                "math": top,
                "math_by_module": by_module,
            }) + "\n")
            written += 1
    print(f"[math_decompose] resolved={resolved} written={written} -> {OUT}")


if __name__ == "__main__":
    main()
