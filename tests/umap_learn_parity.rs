//! Stage-by-stage parity with umap-learn 0.5.12, given the same kNN.
//!
//! `fixtures/gen_umap_learn_fixtures.py` dumps every stage; each test here feeds the same input
//! to the Rust stage and compares.
//!
//! Two copies of the fuzzy-stage fixtures exist. `umap-learn`'s stage functions are numba-
//! compiled for float32 — `compute_membership_strengths` refuses float64 — and that alone puts
//! ~1e-5 of relative noise on a membership's tail. The `_f64` copies come from a line-for-line
//! float64 transcription of the same three functions in the generator, and against those the
//! Rust agrees to 1e-12: the evidence that the arithmetic is identical and the residual on the
//! float32 copies is the reference's precision, not a difference of definition. Both are kept;
//! float32 at 1e-4 documents what a real `umap-learn` run produces.
use umaprs::fuzzy;

const K: usize = 15;
const N: usize = 2000;

fn fixture(name: &str) -> Vec<Vec<f64>> {
    let path = format!(
        "{}/fixtures/umap_learn/{name}.csv",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split(',').map(|v| v.trim().parse().unwrap()).collect())
        .collect()
}

fn flat(rows: &[Vec<f64>]) -> Vec<f64> {
    rows.iter().flatten().copied().collect()
}

fn knn() -> (Vec<usize>, Vec<f64>) {
    let inds: Vec<usize> = flat(&fixture("knn_indices"))
        .iter()
        .map(|v| *v as usize)
        .collect();
    let dists = flat(&fixture("knn_dists"));
    assert_eq!(inds.len(), N * K);
    for i in 0..N {
        assert_eq!(inds[i * K], i, "column 0 is self");
        assert_eq!(dists[i * K], 0.0);
    }
    (inds, dists)
}

fn assert_close(got: f64, want: f64, rel: f64, what: &str) {
    let scale = want.abs().max(1e-12);
    assert!(
        (got - want).abs() / scale <= rel,
        "{what}: got {got:.10e}, umap-learn says {want:.10e} (rel {:.2e})",
        (got - want).abs() / scale
    );
}

fn check_sigmas_rhos(name: &str, rel: f64) {
    let (_, dists) = knn();
    let (sig, rho) = fuzzy::smooth_knn_dist(&dists, N, K, 1.0);
    let want = fixture(name);
    for i in 0..N {
        assert_close(rho[i], want[i][1], rel, &format!("rho[{i}]"));
        assert_close(sig[i], want[i][0], rel, &format!("sigma[{i}]"));
    }
}

#[test]
fn sigmas_and_rhos_match_smooth_knn_dist_in_f64() {
    check_sigmas_rhos("sigmas_rhos_f64", 1e-12);
}

#[test]
fn sigmas_and_rhos_match_a_real_float32_run() {
    check_sigmas_rhos("sigmas_rhos", 1e-4);
}

fn check_memberships(name: &str, rel: f64) {
    let (inds, dists) = knn();
    let (sig, rho) = fuzzy::smooth_knn_dist(&dists, N, K, 1.0);
    let (rows, cols, vals) =
        fuzzy::compute_membership_strengths(&inds, &dists, N, K, &sig, &rho, false);
    let want = fixture(name);
    assert_eq!(want.len(), N * K);
    for (e, w) in want.iter().enumerate() {
        assert_eq!(
            (rows[e], cols[e]),
            (w[0] as usize, w[1] as usize),
            "edge {e}"
        );
        assert_close(vals[e], w[2], rel, &format!("membership {e}"));
    }
}

#[test]
fn memberships_match_compute_membership_strengths_in_f64() {
    check_memberships("memberships_f64", 1e-12);
}

#[test]
fn memberships_match_a_real_float32_run() {
    check_memberships("memberships", 1e-4);
}

fn check_graph(name: &str, rel: f64) {
    let (inds, dists) = knn();
    let r = fuzzy::fuzzy_simplicial_set(&inds, &dists, N, K);
    let want = fixture(name);
    let got: Vec<(usize, usize, f64)> = r.graph.edges().collect();
    assert_eq!(
        got.len(),
        want.len(),
        "edge count (umap-learn nnz after eliminate_zeros)"
    );
    for (e, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(
            (g.0, g.1),
            (w[0] as usize, w[1] as usize),
            "edge {e} position"
        );
        assert_close(g.2, w[2], rel, &format!("edge {e} weight"));
    }
}

#[test]
fn the_symmetric_graph_matches_fuzzy_simplicial_set_in_f64() {
    check_graph("graph_f64", 1e-12);
}

#[test]
fn the_symmetric_graph_matches_a_real_float32_run() {
    check_graph("graph", 1e-4);
}

#[test]
fn the_graph_is_deterministic_across_thread_counts() {
    let (inds, dists) = knn();
    let a: Vec<(usize, usize, f64)> = fuzzy::fuzzy_simplicial_set(&inds, &dists, N, K)
        .graph
        .edges()
        .collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let b: Vec<(usize, usize, f64)> = pool.install(|| {
        fuzzy::fuzzy_simplicial_set(&inds, &dists, N, K)
            .graph
            .edges()
            .collect()
    });
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.0, y.0);
        assert_eq!(x.1, y.1);
        assert_eq!(
            x.2.to_bits(),
            y.2.to_bits(),
            "bit-identical regardless of threads"
        );
    }
}

#[test]
fn ab_params_match_find_ab_params() {
    // (min_dist, spread, a, b) from scipy's curve_fit
    for row in fixture("ab_params") {
        let (md, sp, a, b) = (row[0], row[1], row[2], row[3]);
        let (ga, gb) = umaprs::find_ab_params(md, sp);
        assert_close(ga, a, 1e-4, &format!("a at min_dist {md} spread {sp}"));
        assert_close(gb, b, 1e-4, &format!("b at min_dist {md} spread {sp}"));
    }
}
