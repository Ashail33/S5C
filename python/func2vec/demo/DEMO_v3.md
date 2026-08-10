# func2vec v3 — internal-call decomposition

Third iteration. Same 2140 raw files. What's new is the **decomposition layer**: for each class in vocab we import the library it lives in, walk each of its methods' ASTs, and collect the internal helper functions it calls. Those helpers get injected into the embedding corpus as `COMP:*` tokens so that classes with shared internals get pulled toward the same substrate.

## What changed

1. **`decompose.py`** — imports each class from an installed library (currently sklearn 1.9, xgboost 3.2, lightgbm 4.7, pandas 3.0, scipy 1.17, numpy 2.4), runs `inspect.getsource` on it, parses each method with `ast`, and collects the names of every function/method called from inside. A **document-frequency filter** (max_df=0.15) drops components that appear in more than 15% of classes — those are sklearn scaffolding (`validate_data`, `_fit_context`), not distinctive substrate. Also drops Python built-in exceptions and inherited method names (`fit`, `predict`, `transform`, …).
2. **`augment.py`** — writes `data/sequences_aug.jsonl`. For each decomposed class it appends a synthetic "definition sentence" `[class_token, COMP:c1, COMP:c2, ...]` three times (so `COMP:` tokens easily survive `min_count=3`). With `--splice` it also inlines the top-K components after each class occurrence in the real user-code sequences.
3. **`train.py --input --out-name`** — trains on the augmented corpus and writes `models/func2vec_v3.model` alongside the v2 model, so both stay comparable.

## Corpus and vocab

| metric               | v2     | v3     |
|----------------------|--------|--------|
| trained vocab        | 1668   | **2038** |
| — of which `COMP:*`  | 0      | ~370   |
| classes decomposed   | —      | 153    |
| task-tag silhouette  | 0.205  | 0.177  |

## The main finding — v3 is a *different* view, not a strict improvement

Head-to-head cosine similarities between algorithm sibling pairs, v2 vs v3:

| pair                                                   | v2    | v3    | Δ       |
|--------------------------------------------------------|-------|-------|---------|
| `KMeans` ~ `MiniBatchKMeans`                           | 0.550 | 0.619 | **+0.07** |
| `RandomForestClassifier` ~ `ExtraTreesClassifier`      | 0.729 | 0.743 | **+0.01** |
| `RandomForestClassifier` ~ `GradientBoostingClassifier`| 0.492 | 0.545 | **+0.05** |
| `DecisionTreeClassifier` ~ `DecisionTreeRegressor`     | 0.698 | 0.739 | **+0.04** |
| `XGBClassifier` ~ `XGBRegressor`                       | 0.622 | 0.699 | **+0.08** |
| `SVC` ~ `NuSVC`                                        | 0.823 | 0.749 | −0.08     |
| `LogisticRegression` ~ `LogisticRegressionCV`          | 0.669 | 0.599 | −0.07     |
| `PCA` ~ `TruncatedSVD`                                 | 0.599 | 0.536 | −0.06     |
| `XGBClassifier` ~ `LGBMClassifier`                     | 0.727 | 0.602 | −0.13     |

The pattern is extremely consistent:

- **Pairs that share genuine implementation code get closer.** `MiniBatchKMeans` literally reuses KMeans's `_kmeans_plusplus`, `_validate_center_shape`, `choice` helpers. `DecisionTreeClassifier`/`Regressor` share the same tree engine. `XGBClassifier`/`XGBRegressor` are the same C++ path with a different loss. The COMP substrate they share pulls them together.
- **Pairs that are usage-similar but implementation-different get farther apart.** `XGBClassifier` and `LGBMClassifier` are drop-in substitutes in a script but they're entirely separate C++ codebases; nothing they call internally overlaps. `LogisticRegression` and `LogisticRegressionCV` share an API but the CV variant is a much larger machine. `SVC` and `NuSVC` are mathematically distinct SVM formulations calling different libsvm entry points.

So v3 is an **implementation-similarity lens**, and v2 remains the **usage-similarity lens**. Keep both — the appropriate view depends on what you're asking:

| question                                                  | use  |
|-----------------------------------------------------------|------|
| what algorithm chains after X in real scripts?            | v2   |
| what algorithms should be interchangeable in a pipeline?  | v2   |
| what algorithms share code with X (so a speedup helps both)? | v3 |
| what "primitives" does X actually rely on?                | v3 (via `COMP:*` tokens) |

## The genuinely new capability — algorithms as compositions of primitives

v3 unlocks something v2 could not do. Each class's top `COMP:` neighbors are the internal helpers most strongly associated with it — a decomposition into primitive components:

**`sklearn.cluster.KMeans`**
```
0.783  COMP:_validate_center_shape
0.769  COMP:_is_arraylike_not_scalar
0.708  COMP:_kmeans_plusplus
0.609  COMP:_labels_inertia_threadpool_limit
0.598  COMP:toarray
0.555  COMP:choice
```
The embedding says: KMeans is "the algorithm that does center-shape validation, kmeans++ initialization, threadpooled label-inertia, and random choice sampling."

**`sklearn.ensemble.RandomForestClassifier`**
```
0.761  COMP:rollaxis
0.686  COMP:_set_oob_score_and_attributes
0.662  COMP:_compute_missing_values_in_feature_mask
0.591  COMP:_validate_y_class_weight
0.581  COMP:_average
0.530  COMP:_check_max_features
```
"OOB-score bookkeeping, missing-value handling, y class-weight validation, subsampled feature check" — recognizably the tabular-classifier ensemble spec.

**`sklearn.linear_model.LogisticRegression`** — sklearn 1.9's LR is now array-API aware
```
0.822  COMP:_check_solver
0.739  COMP:_from_estimator
0.733  COMP:setup
0.718  COMP:move_to
0.694  COMP:get_namespace_and_device
0.680  COMP:_check_is_fitted
```
`move_to` and `get_namespace_and_device` reveal the new array-API device-portable code path that's specific to LR in this version.

## Chains under the v3 lens

Same greedy chain, `--model func2vec_v3`. Now the chain follows implementation lineage rather than usage lineage:

**`sklearn.cluster.KMeans` (v3)** — stays inside the KMeans family
```
→ sklearn.cluster.KMeans.get_feature_names_out   (sim=0.696)
→ sklearn.externals.six.moves.cStringIO           (sim=0.827)   ← noise
→ sklearn.cluster.MiniBatchKMeans.fit             (sim=0.868)
→ sklearn.cluster.MiniBatchKMeans.predict         (sim=0.889)
→ sklearn.cluster.MiniBatchKMeans                 (sim=0.905)
```

**`sklearn.tree.DecisionTreeClassifier` (v3)** — walks into the tree/ensemble family
```
→ sklearn.tree.DecisionTreeRegressor             (sim=0.739)
→ sklearn.ensemble.BaggingRegressor.fit          (sim=0.799)
→ sklearn.ensemble.BaggingClassifier.score       (sim=0.861)
→ sklearn.ensemble.BaggingRegressor              (sim=0.823)
→ sklearn.tree.DecisionTreeRegressor.fit         (sim=0.805)
```

**`sklearn.ensemble.RandomForestClassifier` (v3)**
```
→ sklearn.ensemble.ExtraTreesClassifier          (sim=0.743)
→ sklearn.cross_validation.StratifiedKFold       (sim=0.852)
→ sklearn.ensemble.ExtraTreesRegressor           (sim=0.849)
→ sklearn.ensemble.ExtraTreesClassifier.fit      (sim=0.863)
→ sklearn.grid_search.GridSearchCV               (sim=0.844)
```

Compare to the v2 chain for the same seed which stayed within RandomForest itself (`.fit → .score → HistGradientBoostingClassifier.fit`). v3 crosses into ExtraTrees more readily because they literally share tree-building code.

## What still needs work

- **torch / tensorflow / keras decomposition is stubbed** — those packages aren't installed in this session's container (~700MB each). The decomposer is library-agnostic; running it in an environment where they're installed would extend the substrate coverage to CNN/RNN primitives (which would probably help pull `Conv2d`/`Conv1d` even tighter and expose cross-framework structure like `torch.nn.Linear` ~ `keras.layers.Dense` via shared "matmul + bias + activation" substrate).
- **Cross-library substrate is invisible.** The reason `XGBClassifier ~ LGBMClassifier` dropped is that they share zero internal-call names (different codebases). A future decomposer could work at a *semantic primitive* level — "gradient boosting", "tree splitter", "leaf-value optimizer" — instead of raw function names, using a small ontology or a learned mapping.
- **The two-lens result should probably be a two-tower model.** Instead of picking v2 or v3 per query, learn a shared space with two heads (usage / implementation) and expose both nearest-neighbor tables from one model.
