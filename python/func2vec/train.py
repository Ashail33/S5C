"""Train word2vec (skip-gram) on extracted ML call sequences.

Reads data/sequences.jsonl, writes models/func2vec.model plus vectors.tsv/metadata.tsv
compatible with https://projector.tensorflow.org for interactive exploration.

Usage:
    python -m func2vec.train --size 100 --window 5 --min-count 3 --epochs 15
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from gensim.models import Word2Vec

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
SEQ = DATA / "sequences.jsonl"


def iter_sequences(path: Path) -> list[list[str]]:
    seqs: list[list[str]] = []
    with path.open() as f:
        for line in f:
            try:
                seqs.append(json.loads(line)["seq"])
            except Exception:
                continue
    return seqs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=100, help="embedding dim")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--sg", type=int, default=1, help="1=skip-gram, 0=CBOW")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--input", default=str(SEQ), help="path to sequences .jsonl (default: data/sequences.jsonl)")
    ap.add_argument("--out-name", default="func2vec", help="base name for saved model files (models/<name>.model, vectors_<name>.tsv)")
    args = ap.parse_args()

    MODELS.mkdir(parents=True, exist_ok=True)
    seq_path = Path(args.input)
    sequences = iter_sequences(seq_path)
    if not sequences:
        raise SystemExit(f"No sequences found at {seq_path}. Run scrape.py then extract.py first.")

    counts = Counter(tok for seq in sequences for tok in seq)
    print(f"[train] sequences={len(sequences)} raw-vocab={len(counts)} min-count={args.min_count}")

    model = Word2Vec(
        sentences=sequences,
        vector_size=args.size,
        window=args.window,
        min_count=args.min_count,
        sg=args.sg,
        workers=args.workers,
        epochs=args.epochs,
    )
    model_path = MODELS / f"{args.out_name}.model"
    model.save(str(model_path))
    print(f"[train] wrote {model_path} (vocab={len(model.wv)})")

    # projector-friendly export
    suffix = "" if args.out_name == "func2vec" else f"_{args.out_name}"
    vec_path = MODELS / f"vectors{suffix}.tsv"
    meta_path = MODELS / f"metadata{suffix}.tsv"
    with vec_path.open("w") as vf, meta_path.open("w") as mf:
        mf.write("word\tcount\ttop_module\n")
        for word in model.wv.index_to_key:
            vf.write("\t".join(f"{x:.6f}" for x in model.wv[word]) + "\n")
            mf.write(f"{word}\t{counts[word]}\t{word.split('.')[0]}\n")
    print(f"[train] projector export -> {vec_path}, {meta_path}")


if __name__ == "__main__":
    main()
