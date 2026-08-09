"""Chain functions together by greedy nearest-neighbor rollout.

Given seed function(s), predict the next N functions to call. At each step, score
candidates by mean cosine similarity to the current chain (context vector),
excluding tokens already used and, optionally, tokens from a different top-level
module than the seed.

This is a demo of what a func2vec embedding gives you for pipeline synthesis;
it is intentionally simple — a real recommender would condition on task and
avoid stop-word-like utilities (numpy.array, pandas.DataFrame).

Usage:
    python -m func2vec.chain --seed sklearn.decomposition.PCA --steps 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec

ROOT = Path(__file__).parent
MODELS = ROOT / "models"

# tokens we don't want to suggest — too generic / structural
STOPWORDS = {
    "numpy.array", "numpy.asarray", "numpy.zeros", "numpy.ones", "numpy.arange",
    "pandas.DataFrame", "pandas.Series", "pandas.read_csv",
}


def chain(model: Word2Vec, seeds: list[str], steps: int, same_module: bool, k_candidates: int = 30) -> list[tuple[str, float]]:
    for s in seeds:
        if s not in model.wv:
            raise SystemExit(f"seed {s!r} not in vocab")
    chosen = list(seeds)
    out: list[tuple[str, float]] = []
    top_module = seeds[0].split(".")[0]

    for _ in range(steps):
        ctx = np.mean([model.wv[w] for w in chosen], axis=0)
        # rank all vocab by cosine to context
        sims = model.wv.cosine_similarities(ctx, model.wv.vectors)
        order = np.argsort(-sims)
        pick = None
        for idx in order[: k_candidates * 4]:
            tok = model.wv.index_to_key[idx]
            if tok in chosen or tok in STOPWORDS:
                continue
            if same_module and tok.split(".")[0] != top_module:
                continue
            pick = (tok, float(sims[idx]))
            break
        if pick is None:
            break
        chosen.append(pick[0])
        out.append(pick)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", nargs="+", required=True, help="one or more seed function names")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--any-module", action="store_true", help="allow crossing top-level modules")
    args = ap.parse_args()

    model = Word2Vec.load(str(MODELS / "func2vec.model"))
    picks = chain(model, args.seed, steps=args.steps, same_module=not args.any_module)
    print("seed: " + " → ".join(args.seed))
    for tok, sc in picks:
        print(f"  → {tok}   (sim={sc:0.3f})")


if __name__ == "__main__":
    main()
