//! kd-tree for exact kNN in low dimensions (used up to `KDTREE_MAX_DIMS`).
//!
//! Points live in leaf buckets of at most `LEAF_SIZE`, stored contiguously in tree order. Each
//! split is at the median of the widest dimension in that node. A search keeps the k best as a
//! sorted array and the squared distance from the query to the current cell (the incremental
//! "box distance" of Arya & Mount), so far cells are pruned on the whole box, not just the last
//! split plane.
//!
//! **Results are canonical**: the k nearest by `(squared f32 distance, point index)`, ascending.
//! Distances are computed exactly as before (`dist_sq`, f32, dimension order), so they match the
//! previous one-point-per-node tree bit for bit; the only possible difference from it is which of
//! several points at *exactly* the same distance is kept or listed first, which that tree left to
//! its visiting order.

const LEAF_SIZE: usize = 16;

/// Pruning compares a lower bound computed one way with distances computed another; the slack
/// keeps float rounding from pruning a cell that holds a point at exactly the k-th distance.
const PRUNE_SLACK: f32 = 1.0 + 1e-4;

pub struct KdTree {
    nodes: Vec<KdNode>,
    /// Point data in tree order: `data[p * n_dims..]` is the point `perm[p]`.
    data: Vec<f32>,
    /// Tree position → original point index.
    perm: Vec<u32>,
    /// Original point index → tree position.
    pos: Vec<u32>,
    n_dims: usize,
}

struct KdNode {
    /// Tree positions `lo..hi` covered by this node.
    lo: u32,
    hi: u32,
    split_dim: u16,
    split_val: f32,
    /// Children; 0 = leaf (node 0 is the root, never a child).
    left: u32,
    right: u32,
}

/// The k best so far, ascending by `(distance, index)`.
struct Best {
    k: usize,
    items: Vec<(u32, f32)>,
}

impl Best {
    fn new(k: usize) -> Self {
        Self {
            k,
            items: Vec::with_capacity(k + 1),
        }
    }

    /// Distance a candidate must not exceed to matter; infinite until k are held.
    #[inline]
    fn bound(&self) -> f32 {
        if self.items.len() < self.k {
            f32::INFINITY
        } else {
            self.items[self.k - 1].1
        }
    }

    #[inline]
    fn offer(&mut self, idx: u32, dist: f32) {
        if self.k == 0 {
            return;
        }
        if self.items.len() == self.k {
            let (wi, wd) = self.items[self.k - 1];
            if dist > wd || (dist == wd && idx > wi) {
                return;
            }
            self.items.pop();
        }
        let at = self
            .items
            .partition_point(|&(i, d)| d < dist || (d == dist && i < idx));
        self.items.insert(at, (idx, dist));
    }
}

impl KdTree {
    /// Build from flat f32 data (`n_points × n_dims`, row-major).
    pub fn build(data: &[f32], n_points: usize, n_dims: usize) -> Self {
        let mut perm: Vec<u32> = (0..n_points as u32).collect();
        let mut nodes = Vec::with_capacity(2 * n_points / LEAF_SIZE + 1);
        if n_points > 0 {
            build_recursive(&mut nodes, &mut perm, 0, data, n_dims);
        }
        let mut tree_data = Vec::with_capacity(n_points * n_dims);
        for &p in &perm {
            tree_data.extend_from_slice(&data[p as usize * n_dims..][..n_dims]);
        }
        let mut pos = vec![0u32; n_points];
        for (t, &p) in perm.iter().enumerate() {
            pos[p as usize] = t as u32;
        }
        Self {
            nodes,
            data: tree_data,
            perm,
            pos,
            n_dims,
        }
    }

    /// k nearest points to an arbitrary vector (a point *not* in the tree, so nothing is
    /// excluded). Returns (index, squared distance), ascending by distance then index.
    pub fn query(&self, q: &[f32], k: usize) -> Vec<(u32, f32)> {
        self.search(q, u32::MAX, k)
    }

    /// k nearest neighbours of point `q_idx`, itself excluded. Returns (index, squared
    /// distance), ascending by distance then index.
    pub fn knn(&self, q_idx: usize, k: usize) -> Vec<(u32, f32)> {
        let t = self.pos[q_idx] as usize * self.n_dims;
        let q: Vec<f32> = self.data[t..t + self.n_dims].to_vec();
        self.search(&q, q_idx as u32, k)
    }

    fn search(&self, q: &[f32], exclude: u32, k: usize) -> Vec<(u32, f32)> {
        let mut best = Best::new(k);
        if !self.nodes.is_empty() && k > 0 {
            let mut off = vec![0f32; self.n_dims];
            self.search_node(0, q, exclude, &mut best, &mut off, 0.0);
        }
        best.items
    }

    fn search_node(
        &self,
        node_idx: u32,
        q: &[f32],
        exclude: u32,
        best: &mut Best,
        off: &mut [f32],
        box_dist: f32,
    ) {
        let node = &self.nodes[node_idx as usize];
        if node.left == 0 {
            let d = self.n_dims;
            for t in node.lo as usize..node.hi as usize {
                let idx = self.perm[t];
                if idx == exclude {
                    continue;
                }
                let dist = dist_sq(q, &self.data[t * d..(t + 1) * d]);
                best.offer(idx, dist);
            }
            return;
        }
        let dim = node.split_dim as usize;
        let diff = q[dim] - node.split_val;
        let (near, far) = if diff <= 0.0 {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };
        self.search_node(near, q, exclude, best, off, box_dist);

        // The far cell is at least `diff` away along `dim`; replace this dimension's term.
        let old = off[dim];
        let far_dist = box_dist - old * old + diff * diff;
        if far_dist <= best.bound() * PRUNE_SLACK {
            off[dim] = diff;
            self.search_node(far, q, exclude, best, off, far_dist);
            off[dim] = old;
        }
    }
}

/// Build the node for tree positions `lo..lo + perm.len()`; returns its node index.
fn build_recursive(
    nodes: &mut Vec<KdNode>,
    perm: &mut [u32],
    lo: usize,
    data: &[f32],
    n_dims: usize,
) -> u32 {
    let node_idx = nodes.len() as u32;
    let n = perm.len();
    nodes.push(KdNode {
        lo: lo as u32,
        hi: (lo + n) as u32,
        split_dim: 0,
        split_val: 0.0,
        left: 0,
        right: 0,
    });
    if n <= LEAF_SIZE {
        return node_idx;
    }

    // Widest dimension in this node.
    let mut mins = vec![f32::INFINITY; n_dims];
    let mut maxs = vec![f32::NEG_INFINITY; n_dims];
    for &p in perm.iter() {
        let row = &data[p as usize * n_dims..][..n_dims];
        for (j, &v) in row.iter().enumerate() {
            if v < mins[j] {
                mins[j] = v;
            }
            if v > maxs[j] {
                maxs[j] = v;
            }
        }
    }
    let mut dim = 0;
    let mut widest = f32::NEG_INFINITY;
    for j in 0..n_dims {
        let w = maxs[j] - mins[j];
        if w > widest {
            widest = w;
            dim = j;
        }
    }

    // Median along it: left holds values <= split_val, right values >= split_val.
    let mid = n / 2;
    perm.select_nth_unstable_by(mid, |&a, &b| {
        let va = data[a as usize * n_dims + dim];
        let vb = data[b as usize * n_dims + dim];
        va.total_cmp(&vb).then(a.cmp(&b))
    });
    let split_val = data[perm[mid] as usize * n_dims + dim];

    let (l, r) = perm.split_at_mut(mid);
    let left = build_recursive(nodes, l, lo, data, n_dims);
    let right = build_recursive(nodes, r, lo + mid, data, n_dims);
    let node = &mut nodes[node_idx as usize];
    node.split_dim = dim as u16;
    node.split_val = split_val;
    node.left = left;
    node.right = right;
    node_idx
}

#[inline]
fn dist_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// The canonical answer: every other point, ascending by (distance, index).
    fn brute(
        data: &[f32],
        n: usize,
        d: usize,
        q: &[f32],
        exclude: u32,
        k: usize,
    ) -> Vec<(u32, f32)> {
        let mut all: Vec<(u32, f32)> = (0..n as u32)
            .filter(|&i| i != exclude)
            .map(|i| (i, dist_sq(q, &data[i as usize * d..(i as usize + 1) * d])))
            .collect();
        all.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        all.truncate(k);
        all
    }

    fn check(data: &[f32], n: usize, d: usize, k: usize) {
        let tree = KdTree::build(data, n, d);
        for i in 0..n {
            let q = &data[i * d..(i + 1) * d];
            assert_eq!(
                tree.knn(i, k),
                brute(data, n, d, q, i as u32, k),
                "knn of point {i}"
            );
        }
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..50 {
            let q: Vec<f32> = (0..d).map(|_| rng.gen_range(-6.0..6.0)).collect();
            assert_eq!(tree.query(&q, k), brute(data, n, d, &q, u32::MAX, k));
        }
    }

    #[test]
    fn test_kdtree_basic() {
        // 2D points: two clusters
        let data: Vec<f32> = vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1, 10.1, 10.1,
        ];
        let tree = KdTree::build(&data, 8, 2);

        let results = tree.knn(0, 3);
        assert_eq!(results.len(), 3);
        for &(nb, _) in &results {
            assert!(nb < 4, "point 0 neighbor {} should be in cluster A", nb);
        }

        let results = tree.knn(4, 3);
        for &(nb, _) in &results {
            assert!(nb >= 4, "point 4 neighbor {} should be in cluster B", nb);
        }
    }

    #[test]
    fn test_kdtree_exact() {
        // Verify exact results match brute-force
        let data: Vec<f32> = vec![0.0, 0.0, 3.0, 0.0, 1.0, 0.0, 2.0, 0.0, 4.0, 0.0];
        let tree = KdTree::build(&data, 5, 2);

        // Point 0 (0,0): nearest should be 2 (1,0), then 3 (2,0)
        let results = tree.knn(0, 2);
        assert_eq!(results[0].0, 2); // (1,0)
        assert_eq!(results[1].0, 3); // (2,0)
    }

    /// Random mixtures in 1-16 dimensions, many leaves deep, against brute force.
    #[test]
    fn matches_brute_force_on_mixtures() {
        let mut rng = StdRng::seed_from_u64(1);
        for &(n, d) in &[(3000usize, 1usize), (3000, 2), (4000, 8), (2500, 16)] {
            let centres: Vec<Vec<f32>> = (0..6)
                .map(|_| (0..d).map(|_| rng.gen_range(-5.0..5.0)).collect())
                .collect();
            let mut data = Vec::with_capacity(n * d);
            for _ in 0..n {
                let c = &centres[rng.gen_range(0..centres.len())];
                for &m in c {
                    data.push(m + rng.gen_range(-1.0..1.0));
                }
            }
            check(&data, n, d, 14);
        }
    }

    /// Cytometry has exact duplicates (clipped or saturated events) and values on a grid:
    /// ties must resolve by index, the same way every time.
    #[test]
    fn ties_and_duplicates_resolve_by_index() {
        let mut rng = StdRng::seed_from_u64(2);
        let (n, d) = (3000, 4);
        // Integer grid: lots of equal distances, and repeated points.
        let mut data: Vec<f32> = (0..n * d).map(|_| rng.gen_range(0..6) as f32).collect();
        // 200 copies of one point.
        for i in 0..200 {
            for j in 0..d {
                data[i * 7 * d + j] = 2.0;
            }
        }
        check(&data, n, d, 14);
        // All points identical.
        check(&vec![1.5f32; 600 * 3], 600, 3, 14);
    }

    #[test]
    fn small_and_degenerate_inputs() {
        check(&[], 0, 3, 5);
        check(&[1.0, 2.0], 1, 2, 5);
        let data: Vec<f32> = (0..40).map(|i| i as f32).collect();
        check(&data, 20, 2, 25); // k larger than n - 1
    }
}
