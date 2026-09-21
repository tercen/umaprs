#!/usr/bin/env python
"""Stage-by-stage reference outputs from umap-learn 0.5.12, the version Jamie's pipeline pins.

    /tmp/claude-1000/tercenv/bin/python fixtures/gen_umap_learn_fixtures.py

Every stage is dumped *given the same kNN*, so the Rust port can be checked one stage at a time
rather than only on the final embedding, and independently of which kNN backend it uses:

  knn_indices/knn_dists      exact kNN including self at index 0 (umap-learn's convention)
  sigmas_rhos                smooth_knn_dist
  memberships                compute_membership_strengths (directed, self edge = 0)
  graph                      fuzzy_simplicial_set: the symmetric graph, COO sorted by (row, col)
  ab_params                  find_ab_params at several (min_dist, spread)
  epochs_per_sample          the SGD schedule for the pruned graph at n_epochs = 200
  t_*                        the transform stage on held-out points: kNN vs the training set
                             (no self), sigmas/rhos with local_connectivity 0, bipartite
                             memberships, and the weighted-mean init against a fixed embedding

Synthetic data only, %.17g throughout: at 15 digits a nearest-neighbour test can flip.
"""
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from umap.umap_ import (smooth_knn_dist, compute_membership_strengths, fuzzy_simplicial_set,
                        find_ab_params, make_epochs_per_sample, init_graph_transform)
import umap

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "umap_learn")
K = 15
rng = np.random.default_rng(20260921)

def save(name, arr, fmt="%.17g"):
    np.savetxt(os.path.join(OUT, name + ".csv"), np.asarray(arr), delimiter=",", fmt=fmt)

# --- data: four blobs in 8 dims, plus a held-out set for the transform stage ------------------
n_train, n_test, d = 2000, 300, 8
centres = rng.normal(0, 4, size=(4, d))
lab = rng.integers(0, 4, size=n_train + n_test)
X = centres[lab] + rng.normal(0, 1.0, size=(n_train + n_test, d))
X_train, X_test = X[:n_train].astype(np.float64), X[n_train:].astype(np.float64)
save("train", X_train); save("test", X_test); save("train_labels", lab[:n_train], "%d")

# --- fit-side stages, given exact kNN (self first, as umap-learn's own kNN returns it) ---------
nn = NearestNeighbors(n_neighbors=K, algorithm="brute").fit(X_train)
dists, inds = nn.kneighbors(X_train)              # column 0 is self, distance 0
assert (inds[:, 0] == np.arange(n_train)).all()
save("knn_indices", inds, "%d"); save("knn_dists", dists)

sigmas, rhos = smooth_knn_dist(dists.astype(np.float32), float(K), local_connectivity=1.0)
save("sigmas_rhos", np.c_[sigmas, rhos])

rows, cols, vals, _ = compute_membership_strengths(inds, dists.astype(np.float32), sigmas, rhos)
save("memberships", np.c_[rows, cols, vals], fmt=["%d", "%d", "%.17g"])

graph, _, _ = fuzzy_simplicial_set(X_train, K, np.random.RandomState(0), "euclidean",
                                   knn_indices=inds, knn_dists=dists.astype(np.float32))
g = graph.tocoo()
order = np.lexsort((g.col, g.row))
save("graph", np.c_[g.row[order], g.col[order], g.data[order]], fmt=["%d", "%d", "%.17g"])

ab = [(md, sp, *find_ab_params(sp, md)) for md, sp in
      [(0.1, 1.0), (0.01, 1.0), (0.5, 1.0), (0.3, 1.5), (0.0, 1.0), (0.05, 0.8)]]
save("ab_params", np.array(ab))

n_epochs = 200
gp = graph.copy(); gp.data[gp.data < gp.data.max() / float(n_epochs)] = 0.0; gp.eliminate_zeros()
gp = gp.tocoo(); order = np.lexsort((gp.col, gp.row))
eps = make_epochs_per_sample(gp.data[order], n_epochs)
save("epochs_per_sample", np.c_[gp.row[order], gp.col[order], gp.data[order], eps],
     fmt=["%d", "%d", "%.17g", "%.17g"])

# --- transform-side stages on the held-out points ---------------------------------------------
t_dists, t_inds = nn.kneighbors(X_test, n_neighbors=K)   # no self: queries are not training points
save("t_knn_indices", t_inds, "%d"); save("t_knn_dists", t_dists)
# umap-learn's transform uses local_connectivity - 1 = 0, which makes rho = 0 for every query
t_sig, t_rho = smooth_knn_dist(t_dists.astype(np.float32), float(K), local_connectivity=0.0)
save("t_sigmas_rhos", np.c_[t_sig, t_rho])
t_rows, t_cols, t_vals, _ = compute_membership_strengths(t_inds, t_dists.astype(np.float32),
                                                         t_sig, t_rho, bipartite=True)
save("t_memberships", np.c_[t_rows, t_cols, t_vals], fmt=["%d", "%d", "%.17g"])

# a fixed training embedding so the init stage has a concrete input; umap-learn's own fit
emb = umap.UMAP(n_neighbors=K, min_dist=0.1, random_state=42, n_epochs=50).fit_transform(X_train)
save("train_embedding", emb.astype(np.float64))
import scipy.sparse as sp
t_graph = sp.coo_matrix((t_vals, (t_rows, t_cols)), shape=(n_test, n_train)).tocsr()
t_graph.eliminate_zeros()
save("t_init", init_graph_transform(t_graph, emb.astype(np.float32)).astype(np.float64))

print("train", X_train.shape, "test", X_test.shape, "graph nnz", graph.nnz,
      "pruned nnz", len(eps), "ab rows", len(ab))

# --- float64 references -----------------------------------------------------------------------
# umap-learn's stage functions are numba-compiled for float32 (compute_membership_strengths
# refuses float64 outright), so a real run carries ~1e-5 of float32 noise on a membership's
# tail. To show the *arithmetic* is the same, the three functions are transcribed here line for
# line in float64 numpy, and the Rust is held to 1e-12 against these.
def smooth_knn_dist_f64(distances, k, local_connectivity=1.0):
    target = np.log2(k); n = distances.shape[0]
    rho = np.zeros(n); result = np.zeros(n); mean_distances = np.mean(distances)
    for i in range(n):
        lo, hi, mid = 0.0, np.inf, 1.0
        ith = distances[i]; non_zero = ith[ith > 0.0]
        if non_zero.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity)); interpolation = local_connectivity - index
            if index > 0:
                rho[i] = non_zero[index - 1]
                if interpolation > 1e-5: rho[i] += interpolation * (non_zero[index] - non_zero[index - 1])
            else:
                rho[i] = interpolation * non_zero[0]
        elif non_zero.shape[0] > 0:
            rho[i] = np.max(non_zero)
        for _ in range(64):
            psum = 0.0
            for j in range(1, distances.shape[1]):
                d = ith[j] - rho[i]
                psum += np.exp(-(d / mid)) if d > 0 else 1.0
            if abs(psum - target) < 1e-5: break
            if psum > target: hi = mid; mid = (lo + hi) / 2.0
            else:
                lo = mid; mid = mid * 2 if hi == np.inf else (lo + hi) / 2.0
        result[i] = mid
        if rho[i] > 0.0:
            mean_ith = np.mean(ith)
            if result[i] < 1e-3 * mean_ith: result[i] = 1e-3 * mean_ith
        elif result[i] < 1e-3 * mean_distances: result[i] = 1e-3 * mean_distances
    return result, rho

def memberships_f64(inds, dists, sig, rho, bipartite=False):
    n, k = inds.shape; rows, cols, vals = [], [], []
    for i in range(n):
        for j in range(k):
            if not bipartite and inds[i, j] == i: v = 0.0
            elif dists[i, j] - rho[i] <= 0.0 or sig[i] == 0.0: v = 1.0
            else: v = np.exp(-(dists[i, j] - rho[i]) / sig[i])
            rows.append(i); cols.append(inds[i, j]); vals.append(v)
    return np.array(rows), np.array(cols), np.array(vals)

sig64, rho64 = smooth_knn_dist_f64(dists, float(K))
save("sigmas_rhos_f64", np.c_[sig64, rho64])
r64, c64, v64 = memberships_f64(inds, dists, sig64, rho64)
save("memberships_f64", np.c_[r64, c64, v64], fmt=["%d", "%d", "%.17g"])
import scipy.sparse as sp
A = sp.coo_matrix((v64, (r64, c64)), shape=(n_train, n_train)).tocsr(); A.eliminate_zeros()
T = A.transpose().tocsr(); P = A.multiply(T)
G = (A + T - P).tocoo(); G.eliminate_zeros()
o = np.lexsort((G.col, G.row))
save("graph_f64", np.c_[G.row[o], G.col[o], G.data[o]], fmt=["%d", "%d", "%.17g"])
print("float64 references written")
