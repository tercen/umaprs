use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Mutex;

/// Purpose-built HNSW for UMAP kNN search.
///
/// Built in parallel (rayon): the entry point is the highest-level point, fixed before insertion
/// so no global state moves; every neighbour list has its own lock. Neighbour lists use the
/// heuristic selection of Malkov & Yashunin (Alg. 4) on insertion *and* on overflow — with plain
/// nearest-`M` truncation recall@15 was 0.68 on clustered data and did not improve with the beam
/// width (see `examples/knn_recall.rs`). With one thread in the pool the build is sequential and
/// reproducible; with more, insertion order and therefore the graph depend on scheduling.

const M: usize = 16;
const M0: usize = 32;
const EF_CONSTRUCTION: usize = 100;
const EF_SEARCH: usize = 30;
// ML = 1/ln(M) = 1/ln(16)
const ML: f64 = 0.36067376022224085;

/// Min-heap item (closest first)
#[derive(Clone, Copy)]
struct MinItem {
    dist: f32,
    id: u32,
}
impl PartialEq for MinItem {
    fn eq(&self, o: &Self) -> bool {
        self.dist == o.dist
    }
}
impl Eq for MinItem {}
impl PartialOrd for MinItem {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for MinItem {
    fn cmp(&self, o: &Self) -> Ordering {
        o.dist.partial_cmp(&self.dist).unwrap_or(Ordering::Equal)
    }
}

/// Max-heap item (farthest first)
#[derive(Clone, Copy)]
struct MaxItem {
    dist: f32,
    id: u32,
}
impl PartialEq for MaxItem {
    fn eq(&self, o: &Self) -> bool {
        self.dist == o.dist
    }
}
impl Eq for MaxItem {}
impl PartialOrd for MaxItem {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for MaxItem {
    fn cmp(&self, o: &Self) -> Ordering {
        self.dist.partial_cmp(&o.dist).unwrap_or(Ordering::Equal)
    }
}

/// Reusable visited set using generationeration counter — avoids allocating vec![false; n] per search
struct VisitedSet {
    generation: Vec<u32>,
    current: u32,
}

impl VisitedSet {
    fn new(n: usize) -> Self {
        Self {
            generation: vec![0; n],
            current: 0,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.current = self.current.wrapping_add(1);
        if self.current == 0 {
            // Overflow — rare, just clear
            self.generation.fill(0);
            self.current = 1;
        }
    }

    /// Resize for `n` points if the set was made for a different count.
    fn ensure(&mut self, n: usize) {
        if self.generation.len() != n {
            self.generation = vec![0; n];
            self.current = 0;
        }
    }

    #[inline(always)]
    fn visit(&mut self, i: usize) -> bool {
        if self.generation[i] == self.current {
            return false;
        }
        self.generation[i] = self.current;
        true
    }
}

thread_local! {
    /// One visited set per thread, reused across insertions and queries: allocating an n-sized
    /// set per query was O(n) memory traffic for each of n queries.
    static VISITED: RefCell<VisitedSet> = RefCell::new(VisitedSet::new(0));
}

/// Copy a node's neighbour list into `buf` (a plain slice during search, a locked list during
/// build), so the two paths share one `search_layer`.
trait Neighbors: Sync {
    fn read_into(&self, id: u32, buf: &mut Vec<u32>);
}
impl Neighbors for [Vec<u32>] {
    #[inline]
    fn read_into(&self, id: u32, buf: &mut Vec<u32>) {
        buf.clear();
        buf.extend_from_slice(&self[id as usize]);
    }
}
impl Neighbors for [Mutex<Vec<u32>>] {
    #[inline]
    fn read_into(&self, id: u32, buf: &mut Vec<u32>) {
        buf.clear();
        buf.extend_from_slice(&self[id as usize].lock().unwrap());
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

        let max_layer = point_layers.iter().cloned().max().unwrap_or(0);
        // The entry point is the first point of the top level; it is "inserted" by being the
        // entry, everything else links to it or to what descends from it.
        let entry = point_layers
            .iter()
            .position(|&l| l == max_layer)
            .unwrap_or(0) as u32;

        let layers: Vec<Vec<Mutex<Vec<u32>>>> = (0..=max_layer)
            .map(|_| (0..n_points).map(|_| Mutex::new(Vec::new())).collect())
            .collect();

        (0..n_points)
            .into_par_iter()
            .filter(|&id| id as u32 != entry)
            .for_each(|id| {
                VISITED.with(|v| {
                    let mut visited = v.borrow_mut();
                    visited.ensure(n_points);
                    insert_point(
                        id as u32,
                        point_layers[id],
                        &layers,
                        entry,
                        max_layer,
                        &mut visited,
                        dist_fn,
                    );
                });
            });

        let layers = layers
            .into_iter()
            .map(|layer| layer.into_iter().map(|m| m.into_inner().unwrap()).collect())
            .collect();

        Self {
            layers,
            entry,
            max_layer,
            n_points,
        }
    }

    pub fn search<F: Fn(u32, u32) -> f32>(&self, q: u32, k: usize, dist_fn: &F) -> Vec<(u32, f32)> {
        self.search_ef(q, k, EF_SEARCH, dist_fn)
    }

    /// `search` with an explicit beam width. Recall rises with `ef`; so does the cost.
    /// Measured in `examples/knn_recall.rs`.
    pub fn search_ef<F: Fn(u32, u32) -> f32>(
        &self,
        q: u32,
        k: usize,
        ef_search: usize,
        dist_fn: &F,
    ) -> Vec<(u32, f32)> {
        let mut buf = Vec::with_capacity(M0);
        let mut ep = self.entry;
        for l in (1..=self.max_layer).rev() {
            ep = greedy_closest(ep, q, self.layers[l].as_slice(), &mut buf, dist_fn);
        }
        let ef = ef_search.max(k);
        VISITED.with(|v| {
            let mut visited = v.borrow_mut();
            visited.ensure(self.n_points);
            search_layer(
                ep,
                q,
                ef,
                self.layers[0].as_slice(),
                &mut visited,
                &mut buf,
                dist_fn,
            )
            .into_iter()
            .take(k)
            .map(|item| (item.id, item.dist))
            .collect()
        })
    }
}

fn insert_point<F: Fn(u32, u32) -> f32 + Sync>(
    id: u32,
    pl: usize,
    layers: &[Vec<Mutex<Vec<u32>>>],
    entry: u32,
    max_layer: usize,
    visited: &mut VisitedSet,
    dist_fn: &F,
) {
    let mut buf = Vec::with_capacity(M0);
    let mut ep = entry;
    for l in (pl + 1..=max_layer).rev() {
        ep = greedy_closest(ep, id, layers[l].as_slice(), &mut buf, dist_fn);
    }
    for l in (0..=pl.min(max_layer)).rev() {
        let max_conn = if l == 0 { M0 } else { M };
        let layer = layers[l].as_slice();
        let neighbors = search_layer(ep, id, EF_CONSTRUCTION, layer, visited, &mut buf, dist_fn);
        let selected = select_neighbors_heuristic(
            neighbors
                .into_iter()
                .map(|item| (item.id, item.dist))
                .collect(),
            max_conn,
            dist_fn,
        );
        *layer[id as usize].lock().unwrap() = selected.clone();
        for &nb in &selected {
            let mut nbs = layer[nb as usize].lock().unwrap();
            nbs.push(id);
            if nbs.len() > max_conn {
                prune_neighbors(nb, &mut nbs, max_conn, dist_fn);
            }
        }
        if let Some(&first) = selected.first() {
            ep = first;
        }
    }
}

#[inline]
fn greedy_closest<F: Fn(u32, u32) -> f32, L: Neighbors + ?Sized>(
    mut current: u32,
    target: u32,
    layer: &L,
    buf: &mut Vec<u32>,
    dist_fn: &F,
) -> u32 {
    let mut best_dist = dist_fn(current, target);
    loop {
        let mut changed = false;
        layer.read_into(current, buf);
        for &nb in buf.iter() {
            let d = dist_fn(nb, target);
            if d < best_dist {
                best_dist = d;
                current = nb;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    current
}

/// Beam search in one layer; returns up to `ef` results, closest first.
fn search_layer<F: Fn(u32, u32) -> f32, L: Neighbors + ?Sized>(
    entry: u32,
    target: u32,
    ef: usize,
    layer: &L,
    visited: &mut VisitedSet,
    buf: &mut Vec<u32>,
    dist_fn: &F,
) -> Vec<MinItem> {
    visited.reset();
    visited.visit(entry as usize);

    let entry_dist = dist_fn(entry, target);

    let mut candidates = BinaryHeap::new();
    candidates.push(MinItem {
        dist: entry_dist,
        id: entry,
    });

    let mut results = BinaryHeap::<MaxItem>::new();
    results.push(MaxItem {
        dist: entry_dist,
        id: entry,
    });

    while let Some(c) = candidates.pop() {
        let worst = results.peek().map(|r| r.dist).unwrap_or(f32::MAX);
        if results.len() >= ef && c.dist > worst {
            break;
        }

        layer.read_into(c.id, buf);
        for &nb in buf.iter() {
            if !visited.visit(nb as usize) {
                continue;
            }

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

    let mut out: Vec<MinItem> = results
        .into_iter()
        .map(|r| MinItem {
            dist: r.dist,
            id: r.id,
        })
        .collect();
    out.sort_unstable_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
    out
}

fn prune_neighbors<F: Fn(u32, u32) -> f32>(
    node: u32,
    neighbors: &mut Vec<u32>,
    max_conn: usize,
    dist_fn: &F,
) {
    let mut with_dist: Vec<(u32, f32)> = neighbors
        .iter()
        .map(|&nb| (nb, dist_fn(node, nb)))
        .collect();
    with_dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    *neighbors = select_neighbors_heuristic(with_dist, max_conn, dist_fn);
}

/// Keep at most `max_conn` of `candidates` (sorted by distance to the node, ascending): a
/// candidate is taken when it is closer to the node than to every candidate taken before it.
/// If that leaves slots free, the nearest rejected candidates fill them (hnswlib does not do
/// this; it costs nothing and keeps the degree up in sparse regions).
fn select_neighbors_heuristic<F: Fn(u32, u32) -> f32>(
    candidates: Vec<(u32, f32)>,
    max_conn: usize,
    dist_fn: &F,
) -> Vec<u32> {
    if candidates.len() <= max_conn {
        return candidates.into_iter().map(|(id, _)| id).collect();
    }
    let mut kept: Vec<(u32, f32)> = Vec::with_capacity(max_conn);
    let mut rejected: Vec<u32> = Vec::new();
    for (id, d) in candidates {
        if kept.len() >= max_conn {
            break;
        }
        let closer_to_a_kept = kept.iter().any(|&(k, _)| dist_fn(k, id) < d);
        if closer_to_a_kept {
            rejected.push(id);
        } else {
            kept.push((id, d));
        }
    }
    let mut out: Vec<u32> = kept.into_iter().map(|(id, _)| id).collect();
    for id in rejected {
        if out.len() >= max_conn {
            break;
        }
        out.push(id);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_basic() {
        let points: Vec<[f32; 2]> = vec![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
            [10.1, 10.1],
        ];
        let dist_fn = |i: u32, j: u32| -> f32 {
            let a = &points[i as usize];
            let b = &points[j as usize];
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
        };

        let hnsw = Hnsw::build(8, &dist_fn, 42);
        let results = hnsw.search(0, 3, &dist_fn);
        for &(nb, _) in &results {
            assert!(nb < 4);
        }
        let results = hnsw.search(4, 3, &dist_fn);
        for &(nb, _) in &results {
            assert!(nb >= 4);
        }
    }

    /// With one thread in the pool the parallel build is sequential and the graph is the same
    /// run to run; that is what `threads(1)` promises.
    #[test]
    fn one_thread_build_is_reproducible() {
        let n = 3000;
        let mut rng = SmallRng::seed_from_u64(7);
        let points: Vec<[f32; 20]> = (0..n)
            .map(|_| std::array::from_fn(|_| rng.gen_range(-1.0f32..1.0)))
            .collect();
        let dist_fn = |i: u32, j: u32| -> f32 {
            let (a, b) = (&points[i as usize], &points[j as usize]);
            a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let (g1, g2) = pool.install(|| {
            (
                Hnsw::build(n, &dist_fn, 42).layers,
                Hnsw::build(n, &dist_fn, 42).layers,
            )
        });
        assert_eq!(g1, g2);
    }
}
