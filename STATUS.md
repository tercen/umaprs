# umaprs — status of the `faris/umap-learn-parity` branch, night of 2026-09-21

Goal: `~/tercen/goals/2026-09-21-umaprs.md`. This branch is a PR against `main`; nothing here
has been pushed to `main`. Licence: MIT (added on this branch — the crate had none; MIT matches
`tercen-rs`, the SDK it sits beside, and constrains nothing that links it).

## What changed, and how each change is checked

Every stage is held to `umap-learn 0.5.12` — the version Jamie's pipeline pins — **given the
same kNN**, in `tests/umap_learn_parity.rs`, on synthetic fixtures dumped at 17 significant
digits by `fixtures/gen_umap_learn_fixtures.py`. Two tolerances appear throughout and mean
different things: **1e-12** is against a line-for-line float64 transcription of the reference's
functions (the arithmetic is the same); **1e-4** is against a real run (`umap-learn`'s stage
functions are numba-compiled for float32 and `compute_membership_strengths` refuses float64).

| stage | was | now | checked |
|---|---|---|---|
| ρ | forced to 0 — self inserted at distance 0, then `rho = distances[0]` | distance to the first non-zero neighbour: UMAP's local-connectivity assumption | 1e-12 / 1e-4 |
| σ target | `ln(k)` = 2.7 at k = 15 | `log2(k)` = 3.9 — about twice the effective neighbourhood | 1e-12 / 1e-4 |
| σ floor | none; duplicates drove σ → 0 and every other neighbour to membership 0 | `MIN_K_DIST_SCALE` × mean distance | unit test |
| `n_neighbors` | k others plus a synthetic self | k − 1 others plus self, as upstream counts it | fixture kNN |
| symmetrisation | serial `HashMap`, order rescued by a later sort | CSR merge for `A + Aᵀ − A∘Aᵀ`: parallel per row, deterministic by construction | bit-identical across thread counts |
| a, b | a table of three pairs; **92% / 20% off at `min_dist` 0.5**, the R operator's default | the same Levenberg–Marquardt fit `curve_fit` runs | 1e-4 at six (`min_dist`, `spread`) pairs |
| `pow` in the gradient | exponent-bit interpolation, **5.3% worst relative error**, biased the same way every step | exact `powf` | the old function survives only as the test that records that number |
| transform | brute-force scan of the training set, σ = mean of the neighbours' training σ, 30 steps of attraction only | `umap-learn`'s: indexed kNN, per-query σ with ρ = 0, bipartite memberships, weighted-mean init, negative-sampled SGD against the fixed map at α/4 | kNN exact; σ, memberships, init at 1e-12 / 1e-4; deterministic at any thread count |
| seed | HNSW built with a hard-coded 42 | `random_state` everywhere; `transform_seed` on the model (default 42, as upstream) | — |
| threads | rayon's default pool, no control | `threads(n)` on the builder; 0 = rayon's choice, which honours the container quota | — |
| kNN method | kd-tree up to 40 dims | kd-tree up to 16 dims, HNSW above (see below) | — |
| HNSW graph | nearest-`M` pruning, `EF_SEARCH` 30: recall@15 **0.68** on clustered data; serial build | heuristic neighbour selection (Alg. 4), `EF_SEARCH` 100: recall@15 **0.996**; parallel build | `examples/knn_recall.rs`; `one_thread_build_is_reproducible`; envelope below |

## Measured

**Exact `pow` costs nothing.** The same 50k × 32 fit (`examples/time_fit.rs`): **8.86 s** on this
branch against **10.01 s** on the base commit, before the HNSW work. The CSR fuzzy set more than
pays for the transcendental; no bounded fast path was needed. With the parallel index build the
same fit is **5.96 s**.

**The transform's cost was the kd-tree, not the SGD.** Fit on 100k × 40, transform 100k
(`examples/time_transform_stages.rs`): kNN + σ + init **139.4 s**, SGD **0.3 s**. At 40 dims a
kd-tree barely prunes — 1.4 ms a query is close to a full scan — and the fit's kNN pays the same
price. The cutoff moved to 16 dims; above it HNSW with an exact 2k refine.
Same measurement on the HNSW path: kNN + σ + init **16.6 s**, SGD **2.6 s**, whole transform **19.2 s** — seven times faster, and the fit's kNN gains the same way. After the recall fix and the parallel build (below): kNN + σ + init **7.8 s**, whole transform **9.8 s**. Measured at cohort scale (`examples/time_transform.rs 465000 1200000`, 40 dims, 16 cores): **fit 465k in 95.7 s, transform 1.2M in 103.7 s** — three and a half minutes for what the brute-force scan would have needed an hour for.

**Before the index existed**, transforming 400k against 100k took 547 s — and would have been an
hour at Jamie's 1.2M × 465k.

**References on the same 50k × 32 synthetic set, 3 seeds, `n_neighbors` 15, `min_dist` 0.01,
200 epochs**: `umap-learn 0.5.12` **76.8 s** a fit; `uwot 0.2.5` (`n_sgd_threads = 0`, so the seed
means something) **~20 s**.

## The envelope

`scripts/envelope.py` scores each engine at three seeds on the cytometry parameter sweep's own metrics — kNN
purity against the labels, trustworthiness, seed-to-seed neighbour overlap — on a 10,000-cell
subsample. The first synthetic set (well-separated blobs) was useless for this: purity 1.000,
stability 0.013, because inside a homogeneous blob the 2-D neighbours are arbitrary between
seeds. The set now has twelve overlapping populations with sizes varying 20×.

**AML 1% (50,500 cells × 38 markers, real cytometry, two labels), `n_neighbors` 15, `min_dist` 0.01,
200 epochs, three seeds, metrics on a 10,000-cell subsample:**

| engine | seconds / fit | purity | trustworthiness | stability (Jaccard, k=15) | Procrustes agreement |
|---|---|---|---|---|---|
| `umap-learn 0.5.12` | 41.2 | 0.9841 ± 0.0001 | 0.9680 ± 0.0005 | 0.404 | 0.989 |
| **`umaprs` (this branch)** | **5.5** | 0.9841 ± 0.0000 | 0.9678 ± 0.0001 | 0.456 | 0.987 |
| `uwot 0.2.5` | 29.6 | 0.9841 ± 0.0002 | 0.9676 ± 0.0004 | 0.364 | 0.987 |

On real data the branch is inside both references' envelope on purity, trustworthiness and
Procrustes agreement, ahead of both on seed-to-seed neighbour stability, at **7.5× `umap-learn`'s
speed and 5.4× `uwot`'s**.

**Synthetic (50,000 cells × 32 markers, twelve overlapping populations with sizes varying 20×,
same settings):**

| engine | seconds / fit | purity | trustworthiness | stability (Jaccard, k=15) | Procrustes agreement |
|---|---|---|---|---|---|
| `umap-learn 0.5.12` | 36.3 | 0.9976 ± 0.0001 | 0.9356 ± 0.0002 | 0.012 | 0.718 |
| **`umaprs` (this branch)** | **5.5** | 0.9969 ± 0.0001 | 0.9360 ± 0.0002 | 0.012 | 0.994 |
| `umaprs`, exact kNN | 43.4 | 0.9976 ± 0.0001 | 0.9363 ± 0.0003 | 0.013 | 0.994 |
| `uwot 0.2.5` | 32.5 | 0.9975 ± 0.0001 | 0.9348 ± 0.0002 | 0.011 | 0.893 |

Purity is 0.0007 under the references (the remaining 0.4 % of kNN recall); trustworthiness is
level and Procrustes agreement is well ahead. The exact-kNN row is the ceiling: with the same
graph as the references the branch matches them on everything, at the references' cost.

### The kNN recall bug this found

The first synthetic run came back at purity **0.9315 ± 0.0172** and Procrustes **0.312** while the
exact-kNN run scored 0.9976 / 0.994 — so the fault was the approximate graph. `examples/knn_recall.rs`
measures recall@15 against brute force. The HNSW at `EF_SEARCH = 30` found **68 %** of the true
neighbours and, tellingly, raising the beam width did almost nothing (75 % at 400): the index itself
was not navigable. Cause: neighbour lists were pruned to the plain nearest `M`, so on clustered data
every link pointed into the densest spot and a search could not leave a population. Replacing that
with the paper's heuristic selection (Malkov & Yashunin, Alg. 4 — keep a candidate only if it is
closer to the node than to every neighbour already kept; what `hnswlib` does) gives:

| `EF_SEARCH` | recall@15, before | after, serial build | after, parallel build |
|---|---|---|---|
| 30 | 0.684 | 0.984 | 0.983 |
| 60 | 0.711 | 0.995 | 0.994 |
| **100** (new default) | 0.724 | 0.998 | **0.996** |
| 200 | 0.738 | 0.999 | 0.998 |
| 400 | 0.745 | 1.000 | 0.999 |

The exact 2k refine changes nothing at any width (4k gives the same numbers). Applying the
heuristic only at insertion and truncating on overflow is not enough (0.886 at ef 100), so it
runs on both, and that tripled the index build (5 → 15 s at 50k × 32, transform 19 → 54 s).
The build was serial, so it is now parallel — hnswlib's scheme: entry point fixed as the
highest-level point before insertion, one lock per neighbour list — and the per-query visited
set is per thread instead of allocated n-sized per query. On 16 cores: index 15 → 2.3 s, fit
50k × 32 8.9 → **6.0 s**, transform 100k × 100k at 40 dims 19.2 → **9.8 s**, AML fit 14.2 →
**5.5 s**. With one thread in the pool the build is sequential and reproducible
(`one_thread_build_is_reproducible`); with more, the graph depends on scheduling, as the SGD
already did.


## Left as it was, on purpose

- **PCA init above 2,000 points.** Dense `eigh` below, PCA above — `uwot`'s `spca` fallback.
  `umap-learn` runs sparse Lanczos at any size. The parameter sweep found spectral vs PCA immaterial.
- **2-D only.** The optimiser stores `[AtomicU32; 2]`; the R operator has no `n_components`.
  The two integration tests that ask for 3-D failed on `main` before this branch and still do.
- **HogWild in the fit.** Sampling is seeded per chunk; the float summation order varies with
  the thread count. `threads(1)` is bit-repeatable — the README's "reproducible to 1e-10" was
  only ever true single-threaded, and now says so. `umap-learn` has the same contract.
- **`pca` + transform.** With `pca` set, the model keeps the reduced training data but not the
  projection, so `transform` cannot apply it to new points. Not Jamie's configuration.
- **No NNDescent, densMAP, GPU changes, model persistence.** The "triples CSV" format is untouched.

## For the platform, not this crate

**An operator task is booked one CPU** (`task_service.dart`: `cpus = 1.0` unless a cube-query
estimate applies), enforced as a CFS quota. Every parallel stage here runs on one core's worth
of time in production until there is an operator-level CPU booking, the way `memory_model.json`
books memory. Sarno already sizes its rayon pool from the booking; the missing piece is the
declaration.
