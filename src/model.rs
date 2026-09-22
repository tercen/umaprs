use ndarray::Array2;
use rand::RngCore;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::collections::HashMap;

use crate::kdtree::KdTree;

/// Sampling strategy for selecting training subset from full data.
///
/// Future options to consider:
///
/// - **Geometric sketching** (Hie et al., 2019, Cell Systems):
///   Hash points into hypercubes in feature space, pick one per occupied cell.
///   Gives uniform spatial coverage regardless of density — preserves rare
///   populations (e.g., rare stem cells among abundant T-cells).
///   Complexity: O(n) with hashing, but needs tuning of grid resolution.
///
/// - **Max-min diversity**:
///   Greedily pick the point farthest from all already-selected points.
///   Fills gaps in the selection, deterministic, no hyperparameters.
///   Complexity: O(n × m) where m = sample size — can be approximated
///   with kd-tree queries for nearest-selected-point.
#[derive(Clone, Debug)]
pub enum SamplingStrategy {
    /// Random uniform sampling (default). Simple, fast, works well when
    /// populations are roughly balanced.
    Random,
}

/// A fitted UMAP model that can transform new data.
pub struct UmapModel {
    /// Training data (n_train × n_dims)
    pub training_data: Array2<f64>,
    /// Training embedding (n_train × n_components)
    pub embedding: Array2<f64>,
    /// Per-point sigma from smooth kNN distances
    pub sigmas: Vec<f64>,
    /// Per-point rho (distance to nearest neighbor)
    pub rhos: Vec<f64>,
    /// Curve parameter a
    pub a: f64,
    /// Curve parameter b
    pub b: f64,
    /// Number of neighbors
    pub n_neighbors: usize,
    /// Feature names (optional)
    pub feature_names: Option<Vec<String>>,
    /// What the fit was run with, because the transform derives its own schedule from them.
    pub n_epochs: usize,
    pub learning_rate: f64,
    pub negative_sample_rate: f64,
    pub repulsion_strength: f64,
    /// Seed for the transform's negative sampling. `umap-learn`'s `transform_seed`, default 42.
    pub transform_seed: u64,
    /// Thread count the model was fitted with; `transform` uses the same (0 = global pool).
    pub threads: usize,
}

/// The transform's intermediate stages, exposed so each can be checked against `umap-learn`.
pub struct TransformStages {
    /// `n_new × k` neighbours among the training points, and their distances
    pub knn_indices: Array2<usize>,
    pub knn_dists: Array2<f64>,
    pub sigmas: Vec<f64>,
    pub rhos: Vec<f64>,
    /// bipartite memberships, `n_new × k` in row-major order
    pub memberships: Vec<f64>,
    /// the weighted-mean initial positions, `n_new × n_components`
    pub init: Array2<f64>,
}

impl UmapModel {
    /// Sample a training subset from data.
    /// Returns (train_indices, train_data).
    pub fn sample_train(
        data: &Array2<f64>,
        train_fraction: f64,
        strategy: &SamplingStrategy,
        seed: u64,
    ) -> (Vec<usize>, Array2<f64>) {
        let n = data.nrows();
        let n_train = ((n as f64 * train_fraction).ceil() as usize).max(1).min(n);

        let indices = match strategy {
            SamplingStrategy::Random => {
                let mut idx: Vec<usize> = (0..n).collect();
                let mut rng = SmallRng::seed_from_u64(seed);
                idx.shuffle(&mut rng);
                idx.truncate(n_train);
                idx.sort(); // keep order stable for reproducibility
                idx
            }
        };

        let n_dims = data.ncols();
        let mut train = Array2::zeros((indices.len(), n_dims));
        for (i, &orig) in indices.iter().enumerate() {
            for j in 0..n_dims {
                train[[i, j]] = data[[orig, j]];
            }
        }

        (indices, train)
    }

    /// The epoch count `umap-learn` uses for a transform: 100 for a small batch, 30 for a
    /// large one, or a third of the fit's when that was given explicitly.
    pub fn transform_epochs(&self, n_new: usize) -> usize {
        if self.n_epochs > 0 {
            (self.n_epochs / 3).max(1)
        } else if n_new <= 10_000 {
            100
        } else {
            30
        }
    }

    /// Everything before the optimisation — `umap-learn`'s `transform` up to
    /// `init_graph_transform`.
    ///
    /// `smooth_knn_dist` is run with `local_connectivity - 1 = 0`, which makes ρ = 0 for every
    /// query: a query is not its own neighbour, so the local-connectivity offset does not
    /// apply. The σ sum skips column 0 — the nearest training point — because the function
    /// is written for rows whose column 0 is self. That is a quirk of the reference, copied so
    /// the projection is the one `umap-learn` users see.
    pub fn transform_stages(&self, new_data: &Array2<f64>) -> TransformStages {
        crate::in_pool(self.threads, || self.transform_stages_inner(new_data))
    }

    fn transform_stages_inner(&self, new_data: &Array2<f64>) -> TransformStages {
        let expected_dims = self.training_data.ncols();
        if new_data.ncols() != expected_dims {
            let feat_info = match &self.feature_names {
                Some(names) => format!(": {}", names.join(", ")),
                None => String::new(),
            };
            panic!(
                "Input has {} features but model expects {}{}",
                new_data.ncols(),
                expected_dims,
                feat_info
            );
        }
        let n_new = new_data.nrows();
        let k = self.n_neighbors.min(self.training_data.nrows());
        let n_components = self.embedding.ncols();

        let (knn_indices, knn_dists) =
            crate::knn::compute_knn_external(&self.training_data, new_data, k, self.transform_seed);
        let dists_flat: Vec<f64> = knn_dists.iter().copied().collect();
        let inds_flat: Vec<usize> = knn_indices.iter().copied().collect();
        let (sigmas, rhos) = crate::fuzzy::smooth_knn_dist(&dists_flat, n_new, k, 0.0);
        let (_, _, memberships) = crate::fuzzy::compute_membership_strengths(
            &inds_flat,
            &dists_flat,
            n_new,
            k,
            &sigmas,
            &rhos,
            true,
        );

        // `init_graph_transform`: the membership-weighted mean of the neighbours' positions.
        let mut init = Array2::zeros((n_new, n_components));
        for i in 0..n_new {
            let w = &memberships[i * k..(i + 1) * k];
            let wsum: f64 = w.iter().sum();
            if wsum > 0.0 {
                for j in 0..k {
                    let nb = inds_flat[i * k + j];
                    for c in 0..n_components {
                        init[[i, c]] += w[j] / wsum * self.embedding[[nb, c]];
                    }
                }
            } else {
                for c in 0..n_components {
                    init[[i, c]] = f64::NAN;
                }
            }
        }
        TransformStages {
            knn_indices,
            knn_dists,
            sigmas,
            rhos,
            memberships,
            init,
        }
    }

    /// Transform new data points onto the existing embedding — `umap-learn`'s `transform`.
    ///
    /// After [`transform_stages`], the new points are optimised against the **fixed** training
    /// embedding: attraction along their membership edges, repulsion from negative samples
    /// drawn among the training points, learning rate a quarter of the fit's, the usual
    /// `epochs_per_sample` schedule with edges below `max / n_epochs` pruned first. Only the
    /// new point moves (`move_other = false`), so every new point is independent of every
    /// other: the loop is parallel over points with one RNG each, seeded from
    /// `transform_seed` and the point's index, and is therefore deterministic at any thread
    /// count.
    pub fn transform(&self, new_data: &Array2<f64>) -> Array2<f64> {
        crate::in_pool(self.threads, || self.transform_inner(new_data))
    }

    fn transform_inner(&self, new_data: &Array2<f64>) -> Array2<f64> {
        let st = self.transform_stages_inner(new_data);
        let n_new = new_data.nrows();
        let n_train = self.training_data.nrows();
        let k = self.n_neighbors.min(n_train);
        let n_components = self.embedding.ncols();
        let n_epochs = self.transform_epochs(n_new);

        // Prune, then the schedule, both as `umap-learn` does them.
        let max_w = st.memberships.iter().cloned().fold(0.0f64, f64::max);
        let threshold = max_w / n_epochs as f64;
        let eps: Vec<f32> = st
            .memberships
            .iter()
            .map(|&w| {
                if w < threshold || w <= 0.0 {
                    f32::INFINITY
                } else {
                    (max_w / w) as f32
                }
            })
            .collect();
        let neg_rate = self.negative_sample_rate as f32;
        let a = self.a as f32;
        let b = self.b as f32;
        let gamma = self.repulsion_strength as f32;
        let initial_alpha = (self.learning_rate / 4.0) as f32;
        let train: Vec<f32> = self.embedding.iter().map(|&v| v as f32).collect();
        let seed = self.transform_seed;

        let rows: Vec<Vec<f32>> = (0..n_new)
            .into_par_iter()
            .map(|i| {
                let mut cur: Vec<f32> = (0..n_components).map(|c| st.init[[i, c]] as f32).collect();
                if cur.iter().any(|v| v.is_nan()) {
                    return cur;
                }
                let mut rng =
                    SmallRng::seed_from_u64(seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let my_eps = &eps[i * k..(i + 1) * k];
                let mut next_sample: Vec<f32> = my_eps.to_vec();
                let epns: Vec<f32> = my_eps.iter().map(|e| e / neg_rate).collect();
                let mut next_neg: Vec<f32> = epns.clone();
                let mut other = vec![0.0f32; n_components];
                for n in 0..n_epochs {
                    let alpha = initial_alpha * (1.0 - n as f32 / n_epochs as f32);
                    let nf = n as f32;
                    for e in 0..k {
                        if next_sample[e] > nf {
                            continue;
                        }
                        let nb = st.knn_indices[[i, e]];
                        for c in 0..n_components {
                            other[c] = train[nb * n_components + c];
                        }
                        let dist_sq: f32 =
                            cur.iter().zip(&other).map(|(x, y)| (x - y) * (x - y)).sum();
                        let grad_coeff = if dist_sq > 0.0 {
                            (-2.0 * a * b * dist_sq.powf(b - 1.0)) / (a * dist_sq.powf(b) + 1.0)
                        } else {
                            0.0
                        };
                        for c in 0..n_components {
                            cur[c] += clip(grad_coeff * (cur[c] - other[c])) * alpha;
                        }
                        next_sample[e] += my_eps[e];
                        let n_neg = ((nf - next_neg[e]) / epns[e]) as i64;
                        for _ in 0..n_neg.max(0) {
                            let kk = (rng.next_u32() as usize) % n_train;
                            for c in 0..n_components {
                                other[c] = train[kk * n_components + c];
                            }
                            let dist_sq: f32 =
                                cur.iter().zip(&other).map(|(x, y)| (x - y) * (x - y)).sum();
                            let grad_coeff = if dist_sq > 0.0 {
                                (2.0 * gamma * b)
                                    / ((0.001 + dist_sq) * (a * dist_sq.powf(b) + 1.0))
                            } else {
                                0.0
                            };
                            for c in 0..n_components {
                                let g = if grad_coeff > 0.0 {
                                    clip(grad_coeff * (cur[c] - other[c]))
                                } else {
                                    4.0
                                };
                                cur[c] += g * alpha;
                            }
                        }
                        next_neg[e] += n_neg.max(0) as f32 * epns[e];
                    }
                }
                cur
            })
            .collect();

        let mut output = Array2::zeros((n_new, n_components));
        for (i, r) in rows.iter().enumerate() {
            for c in 0..n_components {
                output[[i, c]] = r[c] as f64;
            }
        }
        output
    }

    /// Export model as CSV triples (subject, predicate, object)
    pub fn save_triples_csv(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        writeln!(f, "subject,predicate,object")?;

        let n_train = self.training_data.nrows();
        let n_dims = self.training_data.ncols();
        let n_components = self.embedding.ncols();

        // Model params
        writeln!(f, "model,a,{}", self.a)?;
        writeln!(f, "model,b,{}", self.b)?;
        writeln!(f, "model,n_neighbors,{}", self.n_neighbors)?;
        writeln!(f, "model,n_train,{}", n_train)?;
        writeln!(f, "model,n_dims,{}", n_dims)?;
        writeln!(f, "model,n_components,{}", n_components)?;

        let feat_names: Vec<String> = match &self.feature_names {
            Some(names) => names.clone(),
            None => (0..n_dims).map(|j| format!("f{}", j)).collect(),
        };
        for (j, name) in feat_names.iter().enumerate() {
            writeln!(f, "model,feature_{},{}", j, name)?;
        }

        // Per-point data
        for i in 0..n_train {
            let subj = format!("point_{}", i);
            for c in 0..n_components {
                writeln!(f, "{},umap{},{}", subj, c + 1, self.embedding[[i, c]])?;
            }
            writeln!(f, "{},sigma,{}", subj, self.sigmas[i])?;
            writeln!(f, "{},rho,{}", subj, self.rhos[i])?;
            for j in 0..n_dims {
                writeln!(
                    f,
                    "{},{},{}",
                    subj,
                    feat_names[j],
                    self.training_data[[i, j]]
                )?;
            }
        }

        Ok(())
    }

    /// Load model from CSV triples
    pub fn load_triples_csv(path: &str) -> std::io::Result<Self> {
        use std::io::{BufRead, BufReader};
        let f = std::fs::File::open(path)?;
        let reader = BufReader::new(f);
        let mut triples = Vec::new();
        let mut first = true;
        for line in reader.lines() {
            let line = line?;
            if first {
                first = false;
                continue;
            }
            let parts: Vec<&str> = line.splitn(3, ',').collect();
            if parts.len() == 3 {
                triples.push((
                    parts[0].to_string(),
                    parts[1].to_string(),
                    parts[2].to_string(),
                ));
            }
        }
        Ok(Self::from_triples(&triples))
    }

    fn from_triples(triples: &[(String, String, String)]) -> Self {
        let mut params: HashMap<String, String> = HashMap::new();
        let mut point_data: HashMap<String, HashMap<String, String>> = HashMap::new();

        for (s, p, o) in triples {
            if s == "model" {
                params.insert(p.clone(), o.clone());
            } else if s.starts_with("point_") {
                point_data
                    .entry(s.clone())
                    .or_default()
                    .insert(p.clone(), o.clone());
            }
        }

        let a: f64 = params["a"].parse().unwrap();
        let b: f64 = params["b"].parse().unwrap();
        let n_neighbors: usize = params["n_neighbors"].parse().unwrap();
        let n_train: usize = params["n_train"].parse().unwrap();
        let n_dims: usize = params["n_dims"].parse().unwrap();
        let n_components: usize = params["n_components"].parse().unwrap();

        let mut feature_names = vec![String::new(); n_dims];
        for j in 0..n_dims {
            if let Some(name) = params.get(&format!("feature_{}", j)) {
                feature_names[j] = name.clone();
            }
        }

        let mut training_data = Array2::zeros((n_train, n_dims));
        let mut embedding = Array2::zeros((n_train, n_components));
        let mut sigmas = vec![0.0; n_train];
        let mut rhos = vec![0.0; n_train];

        for i in 0..n_train {
            if let Some(pdata) = point_data.get(&format!("point_{}", i)) {
                for c in 0..n_components {
                    if let Some(v) = pdata.get(&format!("umap{}", c + 1)) {
                        embedding[[i, c]] = v.parse().unwrap();
                    }
                }
                if let Some(v) = pdata.get("sigma") {
                    sigmas[i] = v.parse().unwrap();
                }
                if let Some(v) = pdata.get("rho") {
                    rhos[i] = v.parse().unwrap();
                }
                for j in 0..n_dims {
                    if let Some(v) = pdata.get(&feature_names[j]) {
                        training_data[[i, j]] = v.parse().unwrap();
                    }
                }
            }
        }

        Self {
            training_data,
            embedding,
            sigmas,
            rhos,
            a,
            b,
            n_neighbors,
            feature_names: Some(feature_names),
            n_epochs: 0,
            learning_rate: 1.0,
            negative_sample_rate: 5.0,
            repulsion_strength: 1.0,
            transform_seed: 42,
            threads: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csv_roundtrip() {
        let model = UmapModel {
            training_data: Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap(),
            embedding: Array2::from_shape_vec((3, 2), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap(),
            sigmas: vec![0.5, 0.6, 0.7],
            rhos: vec![0.1, 0.2, 0.3],
            a: 1.577,
            b: 0.8951,
            n_neighbors: 15,
            feature_names: Some(vec!["x".into(), "y".into()]),
            n_epochs: 0,
            learning_rate: 1.0,
            negative_sample_rate: 5.0,
            repulsion_strength: 1.0,
            transform_seed: 42,
            threads: 0,
        };

        let path = "/tmp/umap_test_model.csv";
        model.save_triples_csv(path).unwrap();
        let restored = UmapModel::load_triples_csv(path).unwrap();

        assert_eq!(restored.a, model.a);
        assert_eq!(restored.n_neighbors, model.n_neighbors);
        assert!((restored.embedding[[0, 0]] - 0.1).abs() < 1e-10);
        assert!((restored.sigmas[1] - 0.6).abs() < 1e-10);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_sample_random() {
        let data = Array2::from_shape_vec((100, 3), (0..300).map(|x| x as f64).collect()).unwrap();
        let (indices, train) = UmapModel::sample_train(&data, 0.1, &SamplingStrategy::Random, 42);
        assert_eq!(indices.len(), 10);
        assert_eq!(train.shape(), &[10, 3]);
        // Indices should be sorted and unique
        for i in 1..indices.len() {
            assert!(indices[i] > indices[i - 1]);
        }
    }

    #[test]
    fn test_transform_shape_check() {
        let model = UmapModel {
            training_data: Array2::zeros((5, 3)),
            embedding: Array2::zeros((5, 2)),
            sigmas: vec![1.0; 5],
            rhos: vec![0.0; 5],
            a: 1.577,
            b: 0.8951,
            n_neighbors: 3,
            feature_names: Some(vec!["a".into(), "b".into(), "c".into()]),
            n_epochs: 0,
            learning_rate: 1.0,
            negative_sample_rate: 5.0,
            repulsion_strength: 1.0,
            transform_seed: 42,
            threads: 0,
        };

        let result = std::panic::catch_unwind(|| {
            model.transform(&Array2::zeros((2, 5))) // wrong: 5 features, expects 3
        });
        assert!(result.is_err());
    }
}

#[inline]
fn clip(v: f32) -> f32 {
    v.clamp(-4.0, 4.0)
}
