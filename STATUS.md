# umaprs — status of the `faris/umap-learn-parity` branch, night of 2026-09-21

Goal: `~/tercen/goals/2026-09-21-umaprs.md`. The crate is Alex's; this branch is a PR, and
nothing here has been pushed to `main`.

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

## Measured

**Exact `pow` costs nothing.** The same 50k × 32 fit (`examples/time_fit.rs`): **8.86 s** on this
branch against **10.01 s** on the base commit. The CSR fuzzy set more than pays for the
transcendental; no bounded fast path was needed.

**The transform's cost was the kd-tree, not the SGD.** Fit on 100k × 40, transform 100k
(`examples/time_transform_stages.rs`): kNN + σ + init **139.4 s**, SGD **0.3 s**. At 40 dims a
kd-tree barely prunes — 1.4 ms a query is close to a full scan — and the fit's kNN pays the same
price. The cutoff moved to 16 dims; above it HNSW with an exact 2k refine.
Same measurement on the HNSW path: TODO_HNSW.

**Before the index existed**, transforming 400k against 100k took 547 s — and would have been an
hour at Jamie's 1.2M × 465k.

**References on the same 50k × 32 synthetic set, 3 seeds, `n_neighbors` 15, `min_dist` 0.01,
200 epochs**: `umap-learn 0.5.12` **76.8 s** a fit; `uwot 0.2.5` (`n_sgd_threads = 0`, so the seed
means something) **~20 s**.

## The envelope

`scripts/envelope.py` scores each engine at three seeds on the Lyme sweep's own metrics — kNN
purity against the labels, trustworthiness, seed-to-seed neighbour overlap — on a 10,000-cell
subsample. The first synthetic set (well-separated blobs) was useless for this: purity 1.000,
stability 0.013, because inside a homogeneous blob the 2-D neighbours are arbitrary between
seeds. The set now has twelve overlapping populations with sizes varying 20×.

TODO_ENVELOPE

## Left as it was, on purpose

- **PCA init above 2,000 points.** Dense `eigh` below, PCA above — `uwot`'s `spca` fallback.
  `umap-learn` runs sparse Lanczos at any size. The Lyme sweep found spectral vs PCA immaterial.
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
