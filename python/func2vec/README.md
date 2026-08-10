# func2vec — word2vec for machine-learning functions

Vector embeddings of ML library functions (sklearn, torch, keras, xgboost, ...) trained on their co-occurrence in real Python code scraped from GitHub. The premise is the same as word2vec: functions that appear near each other in similar programs end up near each other in vector space. Once trained, nearest-neighbor lookups suggest which functions typically follow which — useful for pipeline synthesis, code completion, and clustering ML techniques by usage rather than by taxonomy.

## Pipeline

```
scrape.py          →  data/raw/*.py + data/manifest.jsonl
extract.py         →  data/sequences.jsonl (one call sequence per file)
train.py           →  models/func2vec.model (+ vectors.tsv/metadata.tsv)
evaluate.py        →  neighbors, silhouette, models/tsne.png
chain.py           →  greedy pipeline synthesis from a seed function
```

Each stage is idempotent and re-entrant. Re-running `scrape.py` grows the corpus; re-running the rest picks up whatever's on disk. This matters because GitHub code search caps at 1000 results per query, so scaling means running many queries over many sessions.

## Quickstart

```bash
pip install -r requirements.txt

# Mode A — inside a session where the driver has appended hits to data/hits.jsonl
#   (e.g. a Claude session that ran mcp__github__search_code repeatedly)
python -m func2vec.scrape --from-hits

# Mode B — standalone, with a GITHUB_TOKEN that has code-search permission
export GITHUB_TOKEN=...
python -m func2vec.scrape --queries default --pages 5

python -m func2vec.extract --min-len 3
python -m func2vec.train  --size 100 --window 5 --min-count 3 --epochs 15
python -m func2vec.evaluate
python -m func2vec.chain --seed sklearn.decomposition.PCA --steps 5
```

## Method

**Corpus.** GitHub code search finds Python files importing named ML libraries. We fetch the file at its indexed commit (via `raw.githubusercontent.com`) rather than the default branch, so results are reproducible — the sha in the manifest pins the exact source. `scrape.py` maintains `manifest.jsonl` and skips SHAs it already has.

**Extraction.** For each file, `extract.py` builds a symbol table from `import` / `from ... import ...` statements (including aliases like `import numpy as np`, `import torch.nn as nn`), walks the AST in source order, and emits the qualified name of every `ast.Call` whose base name resolves to a whitelisted top-level module. Method calls on instances (`km.fit(X)`) are dropped — resolving those would require type inference. Files with fewer than `--min-len` calls are dropped as too sparse to inform an embedding.

Each sequence is auto-tagged with task labels (`clustering`, `classification`, `regression`, `neural_network`, `preprocessing`, `evaluation`) via keyword heuristics on the function names. The tags are used only for evaluation (silhouette, t-SNE coloring), not for training.

**Training.** gensim `Word2Vec` in skip-gram mode. Defaults: 100-dim vectors, window=5, min_count=3, 15 epochs. Skip-gram is the standard choice when the "vocabulary" is small relative to the corpus and rare tokens matter — here the vocabulary of interesting ML functions is a few thousand, small next to the number of contexts.

**Chaining.** `chain.py` seeds the embedding with one or more known functions, forms a context vector as their mean, and greedily picks the highest-cosine candidate that (a) isn't already in the chain and (b) isn't in a stopword list of generic constructors (`numpy.array`, `pandas.DataFrame`, ...). By default it stays within the seed's top-level module, since crossing modules mid-pipeline usually indicates a poor suggestion; `--any-module` lifts that constraint.

## Demo runs

- **`demo/DEMO.md` (v1)** — first end-to-end run on 1820 files / 1400-token vocab. Established that the pipeline works and surfaced a corpus-contamination problem.
- **`demo/DEMO_v2.md`** — repo blocklist, token stopword filter, and variable→class tracking so `km.fit(X)` resolves to `sklearn.cluster.KMeans.fit`. Vocab 1400 → 1668. Chains started reading like real workflows.
- **`demo/DEMO_v3.md` (current)** — algorithms decomposed into their internal-call substrate via `inspect.getsource` on installed libraries (`decompose.py`), then those components injected into the corpus as `COMP:*` tokens (`augment.py`) and retrained. Vocab 1668 → 2038 (~370 new component tokens). Produces a **second lens** on the same vocabulary: v2 answers "what algorithm typically follows X?" (usage similarity), v3 answers "what algorithm shares implementation substrate with X?" (implementation similarity). E.g. `MiniBatchKMeans` gets pulled toward `KMeans` because they share `_kmeans_plusplus`/`_labels_inertia`; `LGBMClassifier` gets pulled away from `XGBClassifier` because their internals are entirely separate codebases even though scripts use them interchangeably.

## Pipeline (v3)

```
scrape.py     →  data/raw/*.py + data/manifest.jsonl
extract.py    →  data/sequences.jsonl (call sequences per file, w/ method tracking)
decompose.py  →  data/decomposition.jsonl (class → internal components, DF-filtered)
augment.py    →  data/sequences_aug.jsonl (splice + definition sentences)
train.py      →  models/func2vec_v3.model (train --input --out-name to pick corpus/model)
evaluate.py   →  neighbors + task-tag silhouette + t-SNE (--model to pick model)
chain.py      →  greedy pipeline synthesis (--model to pick model; skips COMP: tokens)
```

## Files

- `scrape.py` — corpus collection (two modes: MCP-fed hits or direct code-search)
- `extract.py` — AST-based call-sequence extraction with import resolution
- `train.py` — gensim word2vec training + TF-Projector export
- `evaluate.py` — nearest-neighbors, task-tag silhouette, t-SNE
- `chain.py` — greedy pipeline synthesis by nearest-neighbor rollout
- `_stash_hits.py` — internal helper used to append MCP `search_code` output to `data/hits.jsonl`

## Known limitations

- **Method calls are invisible.** `km.fit(X)` doesn't reach the embedding because we don't do type inference. A meaningful fraction of "what happens next" in an ML pipeline lives on estimator instances, so this is the biggest fidelity loss. A future pass could add lightweight variable→class tracking to recover `fit`/`transform`/`predict`.
- **Static analysis only.** Dead code, `if __name__ == "__main__"` demo blocks, and unused imports all contribute. A runtime-trace corpus would be cleaner but far harder to collect.
- **Task tags are keyword heuristics.** Fine for coloring plots; not a source of ground truth.
- **Corpus is search-biased.** Files chosen by import query are over-representative of specific submodules; well-known repos dominate the top pages.
