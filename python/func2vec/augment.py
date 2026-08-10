"""Augment data/sequences.jsonl with internal-component tokens.

Two augmentations, both driven by data/decomposition.jsonl:

  1. **Definition sentences** — for each decomposed class, append a synthetic
     sequence to the corpus: [class_token, COMP:c1, COMP:c2, ...]. This alone
     is enough to pull classes with shared internal components close to those
     components in embedding space.

  2. **Contextual splice (default off, --splice to enable)** — for every real
     sequence, whenever a class token appears, follow it with its top-K
     component tokens. This propagates the substrate signal into user-code
     contexts too, at the cost of inflating the corpus (~1.5–2×).

Components are prefixed with `COMP:` so they never collide with regular tokens
and can be filtered out at inference time when the user only wants library-facing
tokens back.

Usage:
    python -m func2vec.augment                        # defs only
    python -m func2vec.augment --splice --top 5       # defs + splice
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
SEQ = DATA / "sequences.jsonl"
DECOMP = DATA / "decomposition.jsonl"
OUT = DATA / "sequences_aug.jsonl"

COMP_PREFIX = "COMP:"


def load_decomposition(top: int) -> dict[str, list[str]]:
    if not DECOMP.exists():
        raise SystemExit(f"missing {DECOMP} — run decompose.py first")
    out: dict[str, list[str]] = {}
    with DECOMP.open() as f:
        for line in f:
            r = json.loads(line)
            out[r["class"]] = [f"{COMP_PREFIX}{c}" for c in r["components"][:top]]
    return out


def splice_sequence(seq: list[str], decomp: dict[str, list[str]], top: int) -> list[str]:
    """Insert component tokens after each class occurrence."""
    out: list[str] = []
    for tok in seq:
        out.append(tok)
        if tok in decomp:
            out.extend(decomp[tok][:top])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=8, help="top-K components per class to use")
    ap.add_argument("--splice", action="store_true", help="also splice components into user-code sequences")
    ap.add_argument("--def-repeats", type=int, default=3, help="how many times to emit each definition sentence (raises min-count survival)")
    args = ap.parse_args()

    decomp = load_decomposition(top=args.top)
    print(f"[augment] loaded {len(decomp)} class decompositions", file=sys.stderr)

    n_orig = 0
    n_defs = 0
    n_spliced = 0
    with SEQ.open() as f_in, OUT.open("w") as f_out:
        # 1. Original sequences (optionally spliced)
        for line in f_in:
            try:
                row = json.loads(line)
            except Exception:
                continue
            n_orig += 1
            if args.splice:
                new_seq = splice_sequence(row["seq"], decomp, args.top)
                if len(new_seq) != len(row["seq"]):
                    n_spliced += 1
                row = {**row, "seq": new_seq}
            f_out.write(json.dumps(row) + "\n")

        # 2. Definition sentences (repeated so COMP: tokens easily survive min-count)
        for cls, comps in decomp.items():
            defn = [cls] + comps
            for _ in range(args.def_repeats):
                f_out.write(json.dumps({
                    "file": None,
                    "repo": None,
                    "path": None,
                    "tags": ["_definition"],
                    "seq": defn,
                }) + "\n")
                n_defs += 1

    print(f"[augment] wrote {OUT}: original={n_orig} spliced={n_spliced} defs={n_defs}")


if __name__ == "__main__":
    main()
