use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// Purpose-built HNSW for UMAP kNN search.

const M: usize = 16;
const M0: usize = 32;
const EF_CONSTRUCTION: usize = 100;
const EF_SEARCH: usize = 30;
// ML = 1/ln(M) = 1/ln(16)
const ML: f64 = 0.36067376022224085;

/// Min-heap item (closest first)
#[derive(Clone, Copy)]
struct MinItem { dist: f32, id: u32 }
impl PartialEq for MinItem { fn eq(&self, o: &Self) -> bool { self.dist == o.dist } }
impl Eq for MinItem {}
impl PartialOrd for MinItem { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
impl Ord for MinItem { fn cmp(&self, o: &Self) -> Ordering { o.dist.partial_cmp(&self.dist).unwrap_or(Ordering::Equal) } }

/// Max-heap item (farthest first)
#[derive(Clone, Copy)]
struct MaxItem { dist: f32, id: u32 }
impl PartialEq for MaxItem { fn eq(&self, o: &Self) -> bool { self.dist == o.dist } }
impl Eq for MaxItem {}
impl PartialOrd for MaxItem { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
impl Ord for MaxItem { fn cmp(&self, o: &Self) -> Ordering { self.dist.partial_cmp(&o.dist).unwrap_or(Ordering::Equal) } }

/// Reusable visited set using generationeration counter — avoids allocating vec![false; n] per search
struct VisitedSet {
    generation: Vec<u32>,
    current: u32,
}

impl VisitedSet {
    fn new(n: usize) -> Self { Self { generation: vec![0; n], current: 0 } }

    #[inline(always)]
    fn reset(&mut self) {
        self.current = self.current.wrapping_add(1);
        if self.current == 0 {
            // Overflow — rare, just clear
            self.generation.fill(0);
            self.current = 1;
        }
    }

    #[inline(always)]
    fn visit(&mut self, i: usize) -> bool {
        if self.generation[i] == self.current { return false; }
        self.generation[i] = self.current;
        true
    }
}

pub struct Hnsw {
    layers: Vec<Vec<Vec<u32>>>,
    entry: u32,
    max_layer: usize,
    n_points: usize,
}

impl Hnsw {
    pub fn build<F: Fn(u32, u32) -> f32 + Sync>(n_points: usize, dist_fn: &F, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);

        let point_layers: Vec<usize> = (0..n_points)
            .map(|_| (-rng.gen_range(0.0001f64..1.0).ln() * ML) as usize)
            .collect();

        let overall_max = point_layers.iter().cloned().max().unwrap_or(0);
        let mut layers: Vec<Vec<Vec<u32>>> = (0..=overall_max)
            .map(|_| vec![Vec::new(); n_points])
            .collect();

        let mut max_layer = point_layers[0];
        let mut entry: u32 = 0;
        let mut visited = VisitedSet::new(n_points);

        for id in 1..n_points {
            let id32 = id as u32;
            let pl = point_layers[id];

            let mut ep = entry;

            // Greedy descent from top to pl+1
            for l in (pl.saturating_add(1)..=max_layer).rev() {
                ep = greedy_closest(ep, id32, &layers[l], dist_fn);
            }

            let top = pl.min(max_layer);
            for l in (0..=top).rev() {
                let max_conn = if l == 0 { M0 } else { M };

                let neighbors = search_layer(ep, id32, EF_CONSTRUCTION, &layers[l], &mut visited, dist_fn);

                let selected: Vec<u32> = neighbors.into_iter()
                    .take(max_conn)
                    .map(|item| item.id)
                    .collect();

                layers[l][id] = selected.clone();
                for &nb in &selected {
                    let nbs = &mut layers[l][nb as usize];
                    nbs.push(id32);
                    if nbs.len() > max_conn {
                        prune_neighbors(nb, nbs, max_conn, dist_fn);
                    }
                }

                if let Some(&first) = selected.first() {
                    ep = first;
                }
            }

            if pl > max_layer {
                max_layer = pl;
                entry = id32;
            }
        }

        Self { layers, entry, max_layer, n_points }
    }

    pub fn search<F: Fn(u32, u32) -> f32>(&self, q: u32, k: usize, dist_fn: &F) -> Vec<(u32, f32)> {
        let mut ep = self.entry;

        for l in (1..=self.max_layer).rev() {
            ep = greedy_closest(ep, q, &self.layers[l], dist_fn);
        }

        let ef = EF_SEARCH.max(k);
        // Each thread gets its own visited set via thread_local
        // For the search path we allocate fresh (called from parallel context)
        let mut visited = VisitedSet::new(self.n_points);
        search_layer(ep, q, ef, &self.layers[0], &mut visited, dist_fn)
            .into_iter()
            .take(k)
            .map(|item| (item.id, item.dist))
            .collect()
    }
}

#[inline]
fn greedy_closest<F: Fn(u32, u32) -> f32>(
    mut current: u32, target: u32, layer: &[Vec<u32>], dist_fn: &F,
) -> u32 {
    let mut best_dist = dist_fn(current, target);
    loop {
        let mut changed = false;
        for &nb in &layer[current as usize] {
            let d = dist_fn(nb, target);
            if d < best_dist {
                best_dist = d;
                current = nb;
                changed = true;
            }
        }
        if !changed { break; }
    }
    current
}

/// Search a layer using reusable visited set
fn search_layer<F: Fn(u32, u32) -> f32>(
    entry: u32, target: u32, ef: usize, layer: &[Vec<u32>],
    visited: &mut VisitedSet, dist_fn: &F,
) -> Vec<MinItem> {
    visited.reset();
    visited.visit(entry as usize);

    let entry_dist = dist_fn(entry, target);

    let mut candidates = BinaryHeap::new();
    candidates.push(MinItem { dist: entry_dist, id: entry });

    let mut results = BinaryHeap::<MaxItem>::new();
    results.push(MaxItem { dist: entry_dist, id: entry });

    while let Some(c) = candidates.pop() {
        let worst = results.peek().map(|r| r.dist).unwrap_or(f32::MAX);
        if results.len() >= ef && c.dist > worst {
            break;
        }

        for &nb in &layer[c.id as usize] {
            if !visited.visit(nb as usize) { continue; }

            let d = dist_fn(nb, target);
            let worst = results.peek().map(|r| r.dist).unwrap_or(f32::MAX);

            if results.len() < ef || d < worst {
                candidates.push(MinItem { dist: d, id: nb });
                results.push(MaxItem { dist: d, id: nb });
                if results.len() > ef {
                    results.pop();
                }
            }
        }
    }

    let mut out: Vec<MinItem> = results.into_iter()
        .map(|r| MinItem { dist: r.dist, id: r.id })
        .collect();
    out.sort_unstable_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
    out
}

fn prune_neighbors<F: Fn(u32, u32) -> f32>(
    node: u32, neighbors: &mut Vec<u32>, max_conn: usize, dist_fn: &F,
) {
    let mut with_dist: Vec<(u32, f32)> = neighbors.iter()
        .map(|&nb| (nb, dist_fn(node, nb)))
        .collect();
    with_dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    with_dist.truncate(max_conn);
    *neighbors = with_dist.into_iter().map(|(n, _)| n).collect();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_basic() {
        let points: Vec<[f32; 2]> = vec![
            [0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1],
            [10.0, 10.0], [10.1, 10.0], [10.0, 10.1], [10.1, 10.1],
        ];
        let dist_fn = |i: u32, j: u32| -> f32 {
            let a = &points[i as usize];
            let b = &points[j as usize];
            ((a[0]-b[0]).powi(2) + (a[1]-b[1]).powi(2)).sqrt()
        };

        let hnsw = Hnsw::build(8, &dist_fn, 42);
        let results = hnsw.search(0, 3, &dist_fn);
        for &(nb, _) in &results { assert!(nb < 4); }
        let results = hnsw.search(4, 3, &dist_fn);
        for &(nb, _) in &results { assert!(nb >= 4); }
    }
}
