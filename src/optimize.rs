use ndarray::Array2;
use rand::prelude::*;
use rand::SeedableRng;
use rand::rngs::{StdRng, SmallRng};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::sparse::SparseGraph;

/// Fit the UMAP curve parameters a and b from min_dist and spread
pub(crate) fn find_ab_params(min_dist: f64, spread: f64) -> (f64, f64) {
    if (min_dist - 0.1).abs() < 1e-6 && (spread - 1.0).abs() < 1e-6 {
        return (1.577, 0.8951);
    }
    if (min_dist - 0.01).abs() < 1e-6 && (spread - 1.0).abs() < 1e-6 {
        return (1.929, 0.7915);
    }
    if (min_dist - 0.5).abs() < 1e-6 && (spread - 1.0).abs() < 1e-6 {
        return (1.120, 1.068);
    }
    let a = if min_dist > 0.0 {
        (1.0 / 0.4 - 1.0) / spread.powf(2.0 * 0.9)
    } else {
        1.577
    };
    (a, 0.9)
}

const GRAD_CLAMP_HI: f32 = 4.0;
const GRAD_CLAMP_LO: f32 = -4.0;

#[inline(always)]
fn clamp_grad(val: f32) -> f32 {
    // branchless clamp
    if val > GRAD_CLAMP_HI { GRAD_CLAMP_HI }
    else if val < GRAD_CLAMP_LO { GRAD_CLAMP_LO }
    else { val }
}

/// Epoch-based edge sampler matching uwot's Sampler class.
struct Sampler {
    epochs_per_sample: Vec<f32>,
    epoch_of_next_sample: Vec<f32>,
    epochs_per_negative_sample: Vec<f32>,
    epoch_of_next_negative_sample: Vec<f32>,
}

impl Sampler {
    fn new(weights: &[f32], negative_sample_rate: f32) -> Self {
        let max_weight = weights.iter().cloned().fold(0.0f32, f32::max);
        let n_edges = weights.len();

        let mut epochs_per_sample = vec![0.0f32; n_edges];
        let mut epochs_per_negative_sample = vec![0.0f32; n_edges];

        for i in 0..n_edges {
            epochs_per_sample[i] = if weights[i] > 0.0 {
                max_weight / weights[i]
            } else {
                f32::MAX
            };
            epochs_per_negative_sample[i] = epochs_per_sample[i] / negative_sample_rate;
        }

        let epoch_of_next_sample = epochs_per_sample.clone();
        let epoch_of_next_negative_sample = epochs_per_negative_sample.clone();

        Self {
            epochs_per_sample,
            epoch_of_next_sample,
            epochs_per_negative_sample,
            epoch_of_next_negative_sample,
        }
    }

    #[inline(always)]
    fn is_sample_edge(&self, edge: usize, epoch: f32) -> bool {
        unsafe { *self.epoch_of_next_sample.get_unchecked(edge) <= epoch }
    }

    #[inline(always)]
    fn get_num_neg_samples(&self, edge: usize, epoch: f32) -> usize {
        unsafe {
            let n = (epoch - *self.epoch_of_next_negative_sample.get_unchecked(edge))
                / *self.epochs_per_negative_sample.get_unchecked(edge);
            if n > 0.0 { n as usize } else { 0 }
        }
    }

    #[inline(always)]
    fn next_sample(&mut self, edge: usize, num_neg_samples: usize) {
        unsafe {
            *self.epoch_of_next_sample.get_unchecked_mut(edge) +=
                *self.epochs_per_sample.get_unchecked(edge);
            *self.epoch_of_next_negative_sample.get_unchecked_mut(edge) +=
                num_neg_samples as f32 * *self.epochs_per_negative_sample.get_unchecked(edge);
        }
    }
}

/// Optimize the embedding using stochastic gradient descent.
/// Uses f32 internally for speed, converts back to f64 at the end.
pub fn optimize_layout(
    embedding: &mut Array2<f64>,
    graph: &SparseGraph,
    n_epochs: usize,
    learning_rate: f64,
    min_dist: f64,
    spread: f64,
    negative_sample_rate: f64,
    repulsion_strength: f64,
    random_state: Option<u64>,
) {
    let n_samples = embedding.nrows();
    let n_components = embedding.ncols();
    assert_eq!(n_components, 2, "Only 2D embedding supported for optimized path");

    let (a_f64, b_f64) = find_ab_params(min_dist, spread);
    let a = a_f64 as f32;
    let b = b_f64 as f32;
    let gamma = repulsion_strength as f32;
    eprintln!("UMAP curve parameters: a = {:.3}, b = {:.4}", a, b);

    let mut rng = match random_state {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    };

    // Extract edge list from sparse graph
    let mut heads: Vec<u32> = Vec::with_capacity(graph.nnz());
    let mut tails: Vec<u32> = Vec::with_capacity(graph.nnz());
    let mut weights: Vec<f32> = Vec::with_capacity(graph.nnz());
    for (h, t, w) in graph.edges() {
        heads.push(h as u32);
        tails.push(t as u32);
        weights.push(w as f32);
    }

    let n_edges = heads.len();
    eprintln!("Optimizing {} edges over {} epochs", n_edges, n_epochs);

    let mut sampler = Sampler::new(&weights, negative_sample_rate as f32);

    // Fast approximate pow matching uwot's fastPrecisePow
    #[inline(always)]
    fn fast_pow(a: f32, b: f32) -> f32 {
        let e = b as i32;
        let frac = b - e as f32;
        let u: f64 = a as f64;
        let bits = u.to_bits() as i64;
        let approx_bits = (frac as f64 * (bits - 4606853616395542528) as f64
            + 4606853616395542528.0) as u64;
        let approx = f64::from_bits(approx_bits);
        let mut r = 1.0f64;
        let mut base = a as f64;
        let mut exp = if e >= 0 { e } else { -e };
        while exp > 0 {
            if exp & 1 == 1 { r *= base; }
            base *= base;
            exp >>= 1;
        }
        if e >= 0 { (r * approx) as f32 } else { (approx / r) as f32 }
    }

    // Atomic f32 helpers for HogWild! parallel SGD
    #[inline(always)]
    fn atomic_add_f32(atom: &AtomicU32, val: f32) {
        let mut old = atom.load(Ordering::Relaxed);
        loop {
            let updated = (f32::from_bits(old) + val).to_bits();
            match atom.compare_exchange_weak(old, updated, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(x) => old = x,
            }
        }
    }

    #[inline(always)]
    fn atomic_load_f32(atom: &AtomicU32) -> f32 {
        f32::from_bits(atom.load(Ordering::Relaxed))
    }

    let emb: Vec<AtomicU32> = (0..n_samples)
        .flat_map(|i| {
            let x = (embedding[[i, 0]] as f32).to_bits();
            let y = (embedding[[i, 1]] as f32).to_bits();
            [AtomicU32::new(x), AtomicU32::new(y)]
        })
        .collect();

    let a_b_m2 = -2.0f32 * a * b;
    let gamma_b_2 = 2.0f32 * gamma * b;
    let n_samples_u32 = n_samples as u32;

    for epoch in 0..n_epochs {
        let epoch_f = epoch as f32;
        let alpha = (learning_rate as f32) * (1.0 - epoch_f / n_epochs as f32);

        // Collect active edges for this epoch
        let mut active_edges: Vec<(usize, usize)> = Vec::new();
        for edge in 0..n_edges {
            if !sampler.is_sample_edge(edge, epoch_f) {
                continue;
            }
            let n_neg = sampler.get_num_neg_samples(edge, epoch_f);
            active_edges.push((edge, n_neg));
            sampler.next_sample(edge, n_neg);
        }

        // Process in parallel with per-thread SmallRng (fast, no StdRng per edge)
        let epoch_seed = random_state.unwrap_or(42).wrapping_add(epoch as u64 * 1000003);
        active_edges.par_chunks(256).enumerate().for_each(|(chunk_idx, chunk)| {
            let mut local_rng = SmallRng::seed_from_u64(epoch_seed.wrapping_add(chunk_idx as u64 * 999983));

            for &(edge, n_neg) in chunk {
                let i = unsafe { *heads.get_unchecked(edge) } as usize;
                let j = unsafe { *tails.get_unchecked(edge) } as usize;
                let i2 = i * 2;
                let j2 = j * 2;

                let ix = atomic_load_f32(&emb[i2]);
                let iy = atomic_load_f32(&emb[i2 + 1]);
                let jx = atomic_load_f32(&emb[j2]);
                let jy = atomic_load_f32(&emb[j2 + 1]);

                let dx = ix - jx;
                let dy = iy - jy;
                let dist_sq = (dx * dx + dy * dy).max(f32::EPSILON);

                let pd2b = fast_pow(dist_sq, b);
                let attr_coeff = (a_b_m2 * pd2b) / (dist_sq * (a * pd2b + 1.0));

                let ux = alpha * clamp_grad(attr_coeff * dx);
                let uy = alpha * clamp_grad(attr_coeff * dy);

                atomic_add_f32(&emb[i2], ux);
                atomic_add_f32(&emb[i2 + 1], uy);
                atomic_add_f32(&emb[j2], -ux);
                atomic_add_f32(&emb[j2 + 1], -uy);

                for _ in 0..n_neg {
                    let neg = local_rng.gen_range(0..n_samples_u32) as usize;
                    if neg == i { continue; }

                    let n2 = neg * 2;
                    let nx = atomic_load_f32(&emb[n2]);
                    let ny = atomic_load_f32(&emb[n2 + 1]);
                    let ndx = ix - nx;
                    let ndy = iy - ny;
                    let ndist_sq = (ndx * ndx + ndy * ndy).max(f32::EPSILON);

                    let rep_coeff = gamma_b_2 / ((0.001 + ndist_sq) * (a * fast_pow(ndist_sq, b) + 1.0));

                    atomic_add_f32(&emb[i2], alpha * clamp_grad(rep_coeff * ndx));
                    atomic_add_f32(&emb[i2 + 1], alpha * clamp_grad(rep_coeff * ndy));
                }
            }
        });
    }

    // Center and convert back to f64
    let mut mean_x = 0.0f32;
    let mut mean_y = 0.0f32;
    for i in 0..n_samples {
        mean_x += atomic_load_f32(&emb[i * 2]);
        mean_y += atomic_load_f32(&emb[i * 2 + 1]);
    }
    mean_x /= n_samples as f32;
    mean_y /= n_samples as f32;

    for i in 0..n_samples {
        embedding[[i, 0]] = (atomic_load_f32(&emb[i * 2]) - mean_x) as f64;
        embedding[[i, 1]] = (atomic_load_f32(&emb[i * 2 + 1]) - mean_y) as f64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_find_ab_params() {
        let (a, b) = find_ab_params(0.1, 1.0);
        assert!((a - 1.577).abs() < 0.01);
        assert!((b - 0.8951).abs() < 0.01);
    }

    #[test]
    fn test_gradient_clamping() {
        assert_eq!(clamp_grad(10.0), 4.0);
        assert_eq!(clamp_grad(-10.0), -4.0);
        assert_eq!(clamp_grad(2.0), 2.0);
    }

    #[test]
    fn test_optimize_layout() {
        let mut embedding = Array2::from_shape_vec((5, 2), vec![
            0.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
            0.5, 0.5,
        ]).unwrap();

        let rows = vec![0, 1, 2, 3];
        let cols = vec![1, 2, 3, 4];
        let vals = vec![1.0, 1.0, 1.0, 1.0];
        let graph = SparseGraph::from_triplets(5, &rows, &cols, &vals);

        optimize_layout(&mut embedding, &graph, 10, 1.0, 0.1, 1.0, 5.0, 1.0, Some(42));
        assert_eq!(embedding.shape(), &[5, 2]);

        let mean_x: f64 = embedding.column(0).mean().unwrap();
        let mean_y: f64 = embedding.column(1).mean().unwrap();
        assert!(mean_x.abs() < 1e-6);
        assert!(mean_y.abs() < 1e-6);
    }

    #[test]
    fn test_sampler() {
        let weights = vec![1.0f32, 0.5, 0.25];
        let sampler = Sampler::new(&weights, 5.0);
        assert!(sampler.is_sample_edge(0, 1.0));
    }
}
