/// Quadtree for Barnes-Hut t-SNE O(n log n) repulsion approximation.

const MAX_DEPTH: usize = 50;

pub struct QuadTree {
    nodes: Vec<QuadNode>,
}

struct QuadNode {
    cx: f64, cy: f64, half: f64,
    com_x: f64, com_y: f64, mass: f64,
    children: [u32; 4], // NW, NE, SW, SE; 0 = no child
    is_leaf: bool,
    point_idx: i32, // -1 = empty or subdivided
}

impl QuadTree {
    pub fn build(px: &[f64], py: &[f64]) -> Self {
        let n = px.len();
        if n == 0 { return Self { nodes: vec![] }; }

        let (mut mnx, mut mxx, mut mny, mut mxy) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
        for i in 0..n {
            mnx = mnx.min(px[i]); mxx = mxx.max(px[i]);
            mny = mny.min(py[i]); mxy = mxy.max(py[i]);
        }
        let half = ((mxx - mnx).max(mxy - mny) / 2.0) + 1e-10;
        let cx = (mnx + mxx) / 2.0;
        let cy = (mny + mxy) / 2.0;

        let mut t = Self { nodes: Vec::with_capacity(4 * n) };
        t.nodes.push(QuadNode {
            cx, cy, half, com_x: 0.0, com_y: 0.0, mass: 0.0,
            children: [0; 4], is_leaf: true, point_idx: -1,
        });

        for i in 0..n { t.insert(0, px[i], py[i], i as i32, px, py, 0); }
        t
    }

    fn insert(&mut self, ni: usize, x: f64, y: f64, pidx: i32, px: &[f64], py: &[f64], depth: usize) {
        if depth > MAX_DEPTH { return; }

        let old_m = self.nodes[ni].mass;
        let new_m = old_m + 1.0;
        self.nodes[ni].com_x = (self.nodes[ni].com_x * old_m + x) / new_m;
        self.nodes[ni].com_y = (self.nodes[ni].com_y * old_m + y) / new_m;
        self.nodes[ni].mass = new_m;

        if self.nodes[ni].is_leaf {
            if self.nodes[ni].point_idx < 0 {
                self.nodes[ni].point_idx = pidx;
                return;
            }
            // Subdivide
            let old_pidx = self.nodes[ni].point_idx;
            self.nodes[ni].point_idx = -1;
            self.nodes[ni].is_leaf = false;

            let cx = self.nodes[ni].cx;
            let cy = self.nodes[ni].cy;
            let qh = self.nodes[ni].half / 2.0;
            let off = [(-1.0, 1.0), (1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)];
            for q in 0..4 {
                let ci = self.nodes.len() as u32;
                self.nodes[ni].children[q] = ci;
                self.nodes.push(QuadNode {
                    cx: cx + off[q].0 * qh, cy: cy + off[q].1 * qh, half: qh,
                    com_x: 0.0, com_y: 0.0, mass: 0.0,
                    children: [0; 4], is_leaf: true, point_idx: -1,
                });
            }
            // Re-insert old point
            let opx = px[old_pidx as usize];
            let opy = py[old_pidx as usize];
            let oq = Self::quadrant(cx, cy, opx, opy);
            let oc = self.nodes[ni].children[oq] as usize;
            self.insert(oc, opx, opy, old_pidx, px, py, depth + 1);
        }

        // Insert into correct quadrant
        if !self.nodes[ni].is_leaf {
            let cx = self.nodes[ni].cx;
            let cy = self.nodes[ni].cy;
            let q = Self::quadrant(cx, cy, x, y);
            let c = self.nodes[ni].children[q] as usize;
            self.insert(c, x, y, pidx, px, py, depth + 1);
        }
    }

    fn quadrant(cx: f64, cy: f64, x: f64, y: f64) -> usize {
        if x <= cx { if y > cy { 0 } else { 2 } }
        else { if y > cy { 1 } else { 3 } }
    }

    /// Barnes-Hut repulsion for all points. Returns (fx, fy, Z_normalization).
    pub fn compute_repulsion(&self, px: &[f64], py: &[f64], theta: f64) -> (Vec<f64>, Vec<f64>, f64) {
        let n = px.len();
        let mut fx = vec![0.0; n];
        let mut fy = vec![0.0; n];
        let mut z = 0.0f64;
        for i in 0..n {
            let (fxi, fyi, zi) = self.rep_single(0, px[i], py[i], theta);
            fx[i] = fxi;
            fy[i] = fyi;
            z += zi;
        }
        (fx, fy, z)
    }

    fn rep_single(&self, ni: usize, x: f64, y: f64, theta: f64) -> (f64, f64, f64) {
        if ni >= self.nodes.len() { return (0.0, 0.0, 0.0); }
        let n = &self.nodes[ni];
        if n.mass < 0.5 { return (0.0, 0.0, 0.0); }

        let dx = x - n.com_x;
        let dy = y - n.com_y;
        let d2 = dx * dx + dy * dy;

        // Use cell approximation if far enough or leaf
        if n.is_leaf || (d2 > 1e-10 && (2.0 * n.half) * (2.0 * n.half) < theta * theta * d2) {
            if d2 < 1e-10 && n.mass < 1.5 { return (0.0, 0.0, 0.0); } // skip self
            let q = 1.0 / (1.0 + d2);
            return (n.mass * q * q * dx, n.mass * q * q * dy, n.mass * q);
        }

        let mut fx = 0.0; let mut fy = 0.0; let mut z = 0.0;
        for q in 0..4 {
            let c = n.children[q] as usize;
            if c > 0 {
                let (cfx, cfy, cz) = self.rep_single(c, x, y, theta);
                fx += cfx; fy += cfy; z += cz;
            }
        }
        (fx, fy, z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_and_mass() {
        let x = vec![0.0, 1.0, 5.0, 6.0];
        let y = vec![0.0, 1.0, 5.0, 6.0];
        let t = QuadTree::build(&x, &y);
        assert!((t.nodes[0].mass - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_repulsion_symmetry() {
        let x = vec![-1.0, 1.0];
        let y = vec![0.0, 0.0];
        let t = QuadTree::build(&x, &y);
        let (fx, _, _) = t.compute_repulsion(&x, &y, 0.5);
        // Symmetric points should have opposite forces
        assert!((fx[0] + fx[1]).abs() < 1e-5, "fx={:?}", fx);
    }
}
