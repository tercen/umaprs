#!/usr/bin/env python
"""The envelope comparison: is umaprs inside umap-learn's own seed-to-seed spread?

    /tmp/claude-1000/tercenv/bin/python scripts/envelope.py <name> <data.csv> [labels.csv] [--min-dist 0.01]

For the dataset, runs umap-learn 0.5.12 at three seeds and reads any other engine's embeddings
from results/env_<name>_<engine>_seed<k>.csv (written by `cargo run --release --example
embed_csv` for umaprs and by scripts/envelope_uwot.R for uwot). Scores every embedding on the
metrics the Lyme sweep used:

  purity        fraction of a cell's 15 nearest 2-D neighbours sharing its label (needs labels)
  trust         sklearn trustworthiness, k = 15
  stability     mean Jaccard overlap of 15-NN sets between seeds of the same engine -- floors
                near 0 on homogeneous populations (any 15 of thousands of equidistant cells), so
  procrustes    is reported too: 1 - Procrustes disparity between seeds after the optimal
                rotation/reflection/scale, i.e. how much of the layout is the same map at all

trust and stability are computed on a fixed 10,000-cell subsample so the score is affordable.
A synthetic dataset is available as `<data.csv>` = "synthetic": 50,000 cells x 32 markers in 12
overlapping, unequal Gaussian populations, written to results/ on first use, labels included.
"""
import sys, os, time, json
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import trustworthiness

args = [a for a in sys.argv[1:] if not a.startswith("--")]
name, data_path = args[0], args[1]
labels_path = args[2] if len(args) > 2 else None
md = float(sys.argv[sys.argv.index("--min-dist") + 1]) if "--min-dist" in sys.argv else 0.01
K, SEEDS, SUB = 15, (1, 2, 3), 10_000
OUT = "results"; os.makedirs(OUT, exist_ok=True)

if data_path == "synthetic":
    # Overlapping, unequal populations. Well-separated blobs saturate purity at 1.0 and floor
    # stability (inside a homogeneous blob the 2-D neighbours are arbitrary between seeds), so
    # they cannot tell engines apart. Here neighbouring populations overlap and sizes vary 20x.
    rng = np.random.default_rng(20260921)
    n, d, nb = 50_000, 32, 12
    centres = rng.uniform(-2.2, 2.2, size=(nb, d))
    sizes = rng.dirichlet(np.linspace(0.4, 4, nb)) ; lab = rng.choice(nb, size=n, p=sizes)
    X = centres[lab] + rng.normal(0, 1.0, size=(n, d)) * rng.uniform(0.7, 1.4, size=(nb, 1))[lab]
    data_path, labels_path = f"{OUT}/synthetic_data.csv", f"{OUT}/synthetic_labels.csv"
    np.savetxt(data_path, X, delimiter=",", fmt="%.10g", header=",".join(f"m{i}" for i in range(d)), comments="")
    np.savetxt(labels_path, lab, fmt="%d", header="label", comments="")
X = np.loadtxt(data_path, delimiter=",", skiprows=1)
labels = np.loadtxt(labels_path, delimiter=",", skiprows=1, dtype=str) if labels_path else None
n = X.shape[0]
sub = np.random.default_rng(0).choice(n, size=min(SUB, n), replace=False)

def knn_sets(E):
    nn = NearestNeighbors(n_neighbors=K + 1).fit(E[sub])
    return nn.kneighbors(E[sub], return_distance=False)[:, 1:]

def score(E):
    s = {}
    if labels is not None:
        nbrs = knn_sets(E); ls = labels[sub]
        s["purity"] = float((ls[nbrs] == ls[:, None]).mean())
    s["trust"] = float(trustworthiness(X[sub], E[sub], n_neighbors=K))
    return s

def procrustes_agreement(embs):
    from scipy.spatial import procrustes
    v = []
    for a in range(len(embs)):
        for b in range(a + 1, len(embs)):
            _, _, disparity = procrustes(embs[a][sub], embs[b][sub])
            v.append(1.0 - disparity)
    return float(np.mean(v)) if v else float("nan")

def stability(embs):
    sets = [[set(r) for r in knn_sets(E)] for E in embs]
    j = []
    for a in range(len(sets)):
        for b in range(a + 1, len(sets)):
            j.append(np.mean([len(x & y) / len(x | y) for x, y in zip(sets[a], sets[b])]))
    return float(np.mean(j)) if j else float("nan")

engines = {}
# umap-learn, run here
ul_files = [f"{OUT}/env_{name}_umap-learn_seed{s}.csv" for s in SEEDS]
ul_timing = f"{OUT}/env_{name}_umap-learn_timings.json"
if all(os.path.exists(f) for f in ul_files) and os.path.exists(ul_timing):
    embs = [np.loadtxt(f, delimiter=",") for f in ul_files]; secs = json.load(open(ul_timing))
else:
    import umap
    embs, secs = [], []
    for s in SEEDS:
        t = time.time()
        E = umap.UMAP(n_neighbors=K, min_dist=md, n_epochs=200, random_state=s).fit_transform(X)
        secs.append(time.time() - t); embs.append(E.astype(np.float64))
        np.savetxt(f"{OUT}/env_{name}_umap-learn_seed{s}.csv", E, delimiter=",", fmt="%.8g")
    json.dump(secs, open(ul_timing, "w"))
engines["umap-learn 0.5.12"] = (embs, secs)
# other engines, from files
import glob, re
found = sorted({re.sub(r".*/env_%s_(.+)_seed1\.csv$" % re.escape(name), r"\1", f)
                for f in glob.glob(f"{OUT}/env_{name}_*_seed1.csv")} - {"umap-learn"})
for eng in found:
    files = [f"{OUT}/env_{name}_{eng}_seed{s}.csv" for s in SEEDS]
    if all(os.path.exists(f) for f in files):
        embs = [np.loadtxt(f, delimiter=",") for f in files]
        tf = f"{OUT}/env_{name}_{eng}_timings.json"
        secs = json.load(open(tf)) if os.path.exists(tf) else [float("nan")] * len(SEEDS)
        engines[eng] = (embs, secs)

rows = []
for eng, (embs, secs) in engines.items():
    sc = [score(E) for E in embs]
    row = {"engine": eng, "seconds": f"{np.mean(secs):.1f}",
           "stability": f"{stability(embs):.3f}",
           "procrustes": f"{procrustes_agreement(embs):.3f}",
           "trust": f"{np.mean([c['trust'] for c in sc]):.4f} ± {np.std([c['trust'] for c in sc]):.4f}"}
    if labels is not None:
        row["purity"] = f"{np.mean([c['purity'] for c in sc]):.4f} ± {np.std([c['purity'] for c in sc]):.4f}"
    rows.append(row)
cols = ["engine", "seconds", "purity", "trust", "stability", "procrustes"] if labels is not None else ["engine", "seconds", "trust", "stability", "procrustes"]
print(f"\n### {name} — n = {n}, min_dist = {md}, k = {K}, seeds {SEEDS}, metrics on {len(sub)} cells\n")
print("| " + " | ".join(cols) + " |"); print("|" + "---|" * len(cols))
for r in rows: print("| " + " | ".join(r.get(c, "") for c in cols) + " |")
