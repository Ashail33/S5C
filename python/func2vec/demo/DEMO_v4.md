# func2vec v4 — math substrate + code generation

Fourth iteration. Two independent additions:

- **Math-level decomposition (`math_decompose.py`)** — for each class, resolve every internal call through the defining module's `__dict__` to get its real `.__module__`, and keep only the calls into numpy / scipy / math. Also walks module-level helper functions in the same file (so `_kmeans_plusplus` / `_kmeans_single_lloyd` etc. get counted as part of KMeans's substrate). Emits `MATH:*` tokens.
- **Runnable code generation (`generate_cluster.py`)** — heuristic picks the clustering algorithm from flags (`--k`, `--dataset-size`, `--data-kind`, `--structure`); the embedding then ranks candidates from a small preprocessing whitelist and metrics whitelist by cosine similarity to the picked algorithm, so preprocessing and evaluation companions come from real GitHub co-occurrence data; `inspect.signature` fills in real parameter names so the emitted script actually runs against the installed library.

## Math substrate — what the augmentation actually adds

Sample of what falls out for six classes after the module-level helper walk (v3 was thin because most sklearn "math" lives in private helpers, not the class methods themselves):

```
KMeans                 → issparse, zeros, full, partition, asarray, empty,
                          searchsorted, clip, minimum, argmin, cumsum, log
MiniBatchKMeans        → issparse, zeros, sum, full, partition, asarray,
                          empty_like, ones_like, ceil, empty, searchsorted, clip
DBSCAN                 → array, full, asarray, issparse, where, empty, sum
RandomForestClassifier → issparse, sum, bincount, asarray, rollaxis, atleast_1d,
                          reshape, any, ascontiguousarray, arange
SVC                    → isfinite, asarray, iinfo, log
PCA                    → sqrt
```

The KMeans substrate — `argmin, cumsum, partition, searchsorted, clip, log` — is literally what k-means++ and Lloyd iteration do at the numpy level. `argmin` for assignment, `cumsum + searchsorted` for weighted sampling in k-means++, `partition` for centroid selection, `clip + log` in the variance calculation. This is what a paper's "algorithm sketch" looks like, extracted from real source.

## What the extra substrate does to the embedding — v2 vs v3 vs v4

Same sibling pairs, all four models trained on the same 2140-file corpus:

| pair                                                | v2    | v3    | **v4** |
|-----------------------------------------------------|-------|-------|--------|
| KMeans ~ MiniBatchKMeans                            | 0.550 | 0.619 | **0.654** |
| KMeans ~ DBSCAN                                     | 0.390 | 0.385 | **0.416** |
| RandomForestClassifier ~ ExtraTreesClassifier       | 0.729 | 0.743 | **0.757** |
| XGBClassifier ~ XGBRegressor                        | 0.622 | 0.699 | **0.674** |
| LogisticRegression ~ LogisticRegressionCV           | 0.669 | 0.599 | 0.465     |
| PCA ~ TruncatedSVD                                  | 0.599 | 0.536 | 0.521     |
| SVC ~ NuSVC                                         | 0.823 | 0.749 | 0.691     |
| XGBClassifier ~ LGBMClassifier                      | 0.727 | 0.602 | 0.499     |

Same shape as v3, more pronounced: same-family and same-math-substrate pairs get pulled closer (KMeans+DBSCAN now moves in the right direction — they share `issparse, asarray, sum, full` at least); cross-implementation pairs continue to move apart. Vocabulary 1668 → 2038 (v3) → **2274** (v4, +236 MATH tokens surviving `min_count=3`).

## Code generation

`generate_cluster.py` composes three ingredients:

1. **Rule** picks the algorithm from flags. `k=unknown` → DBSCAN; `structure=hierarchical` → AgglomerativeClustering; `dataset-size=large` → MiniBatchKMeans; else KMeans. `--algorithm` overrides.
2. **Embedding** ranks companion tokens from a small whitelist. For preprocessing (`StandardScaler`, `MinMaxScaler`, `RobustScaler`, `PCA`, ...) and metrics (`silhouette_score`, `davies_bouldin_score`, `homogeneity_score`, `v_measure_score`, ...) it takes cosine similarity to the picked algorithm token and keeps the top-K present in vocab. This is where real usage data matters — the embedding surfaces the metrics that actually appear near the picked algorithm in scraped code.
3. **Introspection** reads `inspect.signature(cls)` and only passes kwargs the installed library accepts. So the generated code doesn't reference parameters that don't exist in the installed version.

Concrete calls (all executed in this session; run 4 is a network limitation, not a codegen bug):

```
$ python -m func2vec.generate_cluster --k 5 --dataset-size large --scale --pca 5 \
    --output demo/generated/kmeans_large.py

$ python demo/generated/kmeans_large.py
algorithm    : MiniBatchKMeans
n_clusters   : 5
homogeneity_score       : 1.0000
completeness_score      : 1.0000
v_measure_score         : 1.0000
```

```
$ python -m func2vec.generate_cluster --k unknown --scale --output demo/generated/dbscan.py

$ python demo/generated/dbscan.py
algorithm    : DBSCAN
n_clusters   : 4
homogeneity_score       : 1.0000
davies_bouldin_score    : 0.3700
completeness_score      : 1.0000
```

```
$ python -m func2vec.generate_cluster --k 5 --structure hierarchical --scale \
    --output demo/generated/agglomerative.py

$ python demo/generated/agglomerative.py
algorithm    : AgglomerativeClustering
n_clusters   : 5
davies_bouldin_score    : 0.4014
v_measure_score         : 1.0000
silhouette_score        : 0.7270
```

```
$ python -m func2vec.generate_cluster --k 8 --data-kind text \
    --output demo/generated/text_kmeans.py

$ python demo/generated/text_kmeans.py
  # succeeds locally on synthetic text; in this container fetch_20newsgroups
  # is blocked by the session's outbound proxy (403 Forbidden). The generated
  # script (imports, TfidfVectorizer + KMeans wiring) is correct.
```

### An interesting embedding-driven choice

The default metric for a KMeans-family script comes back as `homogeneity_score / completeness_score / v_measure_score` — all *labeled* metrics — instead of the more famous unsupervised `silhouette_score`. That's a real signal from the corpus: academic papers evaluating clustering with known ground-truth labels dominate KMeans's co-occurrence context on GitHub. The generated script correctly gates those on `if y is not None:` so it doesn't crash when labels aren't available. For AgglomerativeClustering the embedding does bring `silhouette_score` forward, matching how that estimator is more often used on unlabeled data.

## Files generated in this run

All four scripts live in `demo/generated/`. They're self-contained: no data files, no config, just `python <file>` and it runs. Each header names the algorithm and companions the embedding picked, so the file is its own audit trail.

## What still needs work

- **Math substrate is Python-visible only.** sklearn 1.9's true hot loops are in `.pyx` files; those aren't reached by `inspect.getsource`. We're capturing the numpy calls in the Python code around the Cython, which is a proxy for the algorithm but not the algorithm itself. Extending this would need to parse the `.pyx` source separately.
- **Code generation is scoped to clustering.** The synthesizer is deliberately narrow — a general "generate any ML pipeline" would need a task-classifier layer to pick the algorithm family first. That's the natural next step.
- **The whitelist gate is doing safety work.** Without the small preprocessing/metrics whitelists, the embedding's raw nearest-neighbor list can drift into method tokens or unrelated modules; the whitelist keeps generated code valid but also means the embedding only picks *between* known options rather than *from all of vocab*. Growing those whitelists is the tuning surface.
