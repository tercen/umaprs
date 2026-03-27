use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::Rng;

use crate::codebook::solve_lloyd_max;
use std::sync::OnceLock;

/// Cached Lloyd-Max codebook for N(0,1). Solved once, reused for all encodes.
fn solve_lloyd_max_n01(n_levels: usize) -> (Vec<f32>, Vec<f32>) {
    static CACHE_8: OnceLock<(Vec<f32>, Vec<f32>)> = OnceLock::new();
    static CACHE_128: OnceLock<(Vec<f32>, Vec<f32>)> = OnceLock::new();

    let cache = match n_levels {
        8 => &CACHE_8,
        128 => &CACHE_128,
        _ => return solve_lloyd_max_n01_uncached(n_levels),
    };

    cache.get_or_init(|| solve_lloyd_max_n01_uncached(n_levels)).clone()
}

fn solve_lloyd_max_n01_uncached(n_levels: usize) -> (Vec<f32>, Vec<f32>) {
    // Use large d to get Gaussian limit of Beta distribution
    let gaussian_d = 1024;
    let (raw_c, raw_b) = solve_lloyd_max(gaussian_d, n_levels);
    let scale = (gaussian_d as f32).sqrt();
    let centroids: Vec<f32> = raw_c.iter().map(|&c| c * scale).collect();
    let boundaries: Vec<f32> = raw_b.iter().map(|&b| b * scale).collect();
    (centroids, boundaries)
}

/// TurboQuant-inspired vector quantization for fast approximate distance computation.
///
/// Algorithm:
/// 1. Normalize vectors to unit length (store norms separately)
/// 2. Apply randomized Hadamard transform (random sign flips + Walsh-Hadamard)
/// 3. Scalar quantize each coordinate to b bits using Lloyd-Max codebook
/// 4. For distance computation: dequantize to f32
///
/// This preserves distances with near-optimal distortion guarantees.

/// Quantize a scalar value to the nearest centroid index
#[inline]
fn quantize_scalar_dynamic(val: f32, boundaries: &[f32]) -> u8 {
    let mut idx = 0u8;
    for (i, &b) in boundaries.iter().enumerate() {
        if val > b { idx = (i + 1) as u8; } else { break; }
    }
    idx
}

/// Dequantize a 4-bit index back to f32
#[inline]
fn dequantize_scalar(idx: u8, centroids: &[f32; 16]) -> f32 {
    centroids[idx as usize]
}

/// In-place Walsh-Hadamard Transform (fast, O(n log n))
fn walsh_hadamard_transform(data: &mut [f32]) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());

    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let x = data[j];
                let y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
        }
        h *= 2;
    }

    // Normalize
    let norm = (n as f32).sqrt();
    for v in data.iter_mut() {
        *v /= norm;
    }
}

/// Number of bits per coordinate
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QuantBits {
    Four,
    Eight,
}

/// TurboQuant_prod: (b-1)-bit MSE + 1-bit QJL sign per coordinate.
/// Total bits per coordinate = b (same budget, better inner products).
///
/// Following the paper exactly:
///   Stage 1: Random rotation (Hadamard + signs) → (b-1)-bit Lloyd-Max per coordinate
///   Stage 2: QJL on residual: q = sign(S · r) where S is a separate random sign matrix
///
/// Packing: nibble/byte = (MSE index << 1) | qjl_sign_bit
/// Inner product: dot ≈ mse_dot + (π/2)/d · ||r_i|| · ||r_j|| · sign_agreement
pub struct QuantizedData {
    pub n_samples: usize,
    pub n_dims: usize,
    padded_dims: usize,
    bits: QuantBits,
    /// Random sign flips for Hadamard randomization (Stage 1 rotation)
    signs: Vec<f32>,
    /// Norms of original vectors
    norms: Vec<f32>,
    /// Packed data: each coordinate has (b-1) MSE bits + 1 QJL sign bit
    packed: Vec<u8>,
    /// Lloyd-Max codebook centroids solved for exact Beta(d) distribution
    centroids: Vec<f32>,
    /// Lloyd-Max decision boundaries
    boundaries: Vec<f32>,
    /// QJL: random signs for second Hadamard rotation (Stage 2 projection)
    /// S = diag(qjl_signs) · WHT — orthogonal, S·S^T = I exactly
    qjl_rotation_signs: Vec<f32>,
    /// QJL: global ||r||² per coordinate (constant approximation after WHT)
    qjl_r_norm_sq_per_coord: f32,
}


impl QuantizedData {
    /// Quantize with 4-bit TurboQuant_prod (3-bit MSE + 1-bit QJL sign)
    pub fn encode(data: &Array2<f64>, seed: u64) -> Self {
        Self::encode_with_bits(data, seed, QuantBits::Four)
    }

    /// Quantize with TurboQuant_prod: (b-1)-bit MSE + 1-bit QJL per coordinate
    pub fn encode_with_bits(data: &Array2<f64>, seed: u64, bits: QuantBits) -> Self {
        let n_samples = data.nrows();
        let n_dims = data.ncols();
        let padded_dims = n_dims.next_power_of_two();

        // Lloyd-Max codebook for N(0,1) — matches our coordinate distribution
        // after WHT + √d scaling. Solved at build time via gauss-quad for the
        // exact Beta distribution in the Gaussian limit.
        // The codebook is the same for all d since our scaling normalizes to N(0,1).
        let (centroids, boundaries) = solve_lloyd_max_n01(match bits {
            QuantBits::Four => 8,   // 3-bit MSE = 8 levels
            QuantBits::Eight => 128, // 7-bit MSE = 128 levels
        });

        // Stage 1 rotation: random sign flips for Hadamard
        let mut rng = StdRng::seed_from_u64(seed);
        let signs: Vec<f32> = (0..padded_dims)
            .map(|_| { let v: f32 = rng.gen_range(0.0..1.0); if v < 0.5 { 1.0 } else { -1.0 } })
            .collect();

        // Stage 2 QJL: second randomized Hadamard as orthogonal projection S
        // Different random signs than Stage 1, same WHT → orthogonal, S·S^T = I exactly
        let mut qjl_rng = StdRng::seed_from_u64(seed.wrapping_add(0x514A4C));
        let qjl_signs: Vec<f32> = (0..padded_dims)
            .map(|_| { let v: f32 = qjl_rng.gen_range(0.0..1.0); if v < 0.5 { 1.0 } else { -1.0 } })
            .collect();

        let mut norms = Vec::with_capacity(n_samples);
        let mut total_r_sq = 0.0f64;
        let bytes_per_point = match bits {
            QuantBits::Four => padded_dims / 2,
            QuantBits::Eight => padded_dims,
        };
        let mut packed = Vec::with_capacity(n_samples * bytes_per_point);

        for i in 0..n_samples {
            // Normalize to unit sphere
            let mut vec = vec![0.0f32; padded_dims];
            for j in 0..n_dims {
                vec[j] = data[[i, j]] as f32;
            }
            let norm: f32 = vec.iter().map(|&v| v * v).sum::<f32>().sqrt();
            norms.push(norm);
            if norm > f32::EPSILON {
                for v in vec.iter_mut() { *v /= norm; }
            }

            // Stage 1: Hadamard rotation
            for (v, &s) in vec.iter_mut().zip(signs.iter()) { *v *= s; }
            walsh_hadamard_transform(&mut vec);
            let scale = (padded_dims as f32).sqrt();
            for v in vec.iter_mut() { *v *= scale; }

            // Stage 1: MSE quantize + compute residual per coordinate
            let mut mse_indices = vec![0u8; padded_dims];
            let mut residual = vec![0.0f32; padded_dims];

            match bits {
                QuantBits::Four => {
                    for j in 0..padded_dims {
                        let idx = quantize_scalar_dynamic(vec[j], &boundaries);
                        mse_indices[j] = idx;
                        residual[j] = vec[j] - centroids[idx as usize];
                        total_r_sq += (residual[j] * residual[j]) as f64;
                    }
                }
                QuantBits::Eight => {
                    for j in 0..padded_dims {
                        let idx = quantize_scalar_dynamic(vec[j], &boundaries);
                        mse_indices[j] = idx;
                        residual[j] = vec[j] - centroids[idx as usize];
                        total_r_sq += (residual[j] * residual[j]) as f64;
                    }
                }
            }

            // Stage 2: QJL — project residual through randomized Hadamard S
            // S = diag(qjl_signs) · WHT — orthogonal, so S·S^T = I exactly
            // q = sign(S · residual)
            let mut projected = residual.clone();
            for (v, &s) in projected.iter_mut().zip(qjl_signs.iter()) { *v *= s; }
            walsh_hadamard_transform(&mut projected);

            let mut qjl_sign_bits = vec![0u8; padded_dims];
            for j in 0..padded_dims {
                qjl_sign_bits[j] = if projected[j] >= 0.0 { 1 } else { 0 };
            }

            // Pack: (MSE index << 1) | qjl_sign
            match bits {
                QuantBits::Four => {
                    for j in (0..padded_dims).step_by(2) {
                        let hi = (mse_indices[j] << 1) | qjl_sign_bits[j];
                        let lo = (mse_indices[j + 1] << 1) | qjl_sign_bits[j + 1];
                        packed.push((hi << 4) | lo);
                    }
                }
                QuantBits::Eight => {
                    for j in 0..padded_dims {
                        packed.push((mse_indices[j] << 1) | qjl_sign_bits[j]);
                    }
                }
            }
        }

        let total_coords = (n_samples * padded_dims) as f64;
        let mse_per_coord = (total_r_sq / total_coords) as f32;

        Self {
            n_samples,
            n_dims,
            padded_dims,
            bits,
            signs,
            norms,
            packed,
            centroids,
            boundaries,
            qjl_rotation_signs: qjl_signs,
            qjl_r_norm_sq_per_coord: mse_per_coord,
        }
    }


    /// Dequantize a single vector to f32 (MSE part only, ignoring QJL sign)
    pub fn decode(&self, idx: usize) -> Vec<f32> {
        let bpp = self.bytes_per_point();
        let offset = idx * bpp;

        let mut vec = vec![0.0f32; self.padded_dims];
        match self.bits {
            QuantBits::Four => {
                for j in 0..self.padded_dims / 2 {
                    let byte = self.packed[offset + j];
                    let hi_idx = ((byte >> 4) & 0x0F) >> 1;
                    let lo_idx = (byte & 0x0F) >> 1;
                    vec[j * 2] = self.centroids[hi_idx as usize];
                    vec[j * 2 + 1] = self.centroids[lo_idx as usize];
                }
            }
            QuantBits::Eight => {
                for j in 0..self.padded_dims {
                    let idx = self.packed[offset + j] >> 1;
                    vec[j] = self.centroids[idx as usize];
                }
            }
        }

        // Undo scaling
        let scale = (self.padded_dims as f32).sqrt();
        for v in vec.iter_mut() {
            *v /= scale;
        }

        // Inverse Walsh-Hadamard (WHT is its own inverse up to normalization)
        walsh_hadamard_transform(&mut vec);

        // Undo sign flips
        for (v, &s) in vec.iter_mut().zip(self.signs.iter()) {
            *v *= s;
        }

        // Restore norm
        let norm = self.norms[idx];
        for v in vec.iter_mut() {
            *v *= norm;
        }

        vec[..self.n_dims].to_vec()
    }

    /// Compute approximate squared Euclidean distance between two quantized vectors.
    /// Works directly on quantized representation without full dequantization.
    /// Approximate squared Euclidean distance with QJL-corrected inner product.
    /// MSE dot product + sign agreement correction, all from the packed data.
    pub fn approx_dist_sq(&self, i: usize, j: usize) -> f32 {
        let ni = self.norms[i];
        let nj = self.norms[j];

        let (mse_dot, sign_disagree) = match self.bits {
            QuantBits::Four => {
                let bpp = self.padded_dims / 2;
                let off_i = i * bpp;
                let off_j = j * bpp;
                let mut dot = 0.0f32;
                let mut disagree = 0u32;
                for k in 0..bpp {
                    let bi = self.packed[off_i + k];
                    let bj = self.packed[off_j + k];
                    // Hi nibble: (3-bit idx << 1) | sign
                    let idx_i_hi = ((bi >> 4) & 0x0F) >> 1;
                    let idx_j_hi = ((bj >> 4) & 0x0F) >> 1;
                    let sign_i_hi = (bi >> 4) & 1;
                    let sign_j_hi = (bj >> 4) & 1;
                    dot += self.centroids[idx_i_hi as usize] * self.centroids[idx_j_hi as usize];
                    disagree += (sign_i_hi ^ sign_j_hi) as u32;
                    // Lo nibble
                    let idx_i_lo = (bi & 0x0F) >> 1;
                    let idx_j_lo = (bj & 0x0F) >> 1;
                    let sign_i_lo = bi & 1;
                    let sign_j_lo = bj & 1;
                    dot += self.centroids[idx_i_lo as usize] * self.centroids[idx_j_lo as usize];
                    disagree += (sign_i_lo ^ sign_j_lo) as u32;
                }
                (dot, disagree)
            }
            QuantBits::Eight => {
                let bpp = self.padded_dims;
                let off_i = i * bpp;
                let off_j = j * bpp;
                let mut dot = 0.0f32;
                let mut disagree = 0u32;
                for k in 0..bpp {
                    let bi = self.packed[off_i + k];
                    let bj = self.packed[off_j + k];
                    let idx_i = bi >> 1;
                    let idx_j = bj >> 1;
                    dot += self.centroids[idx_i as usize] * self.centroids[idx_j as usize];
                    disagree += ((bi ^ bj) & 1) as u32;
                }
                (dot, disagree)
            }
        };

        // QJL correction for inner product of two reconstructions:
        //   <x̃_qjl_x, x̃_qjl_y> = (√(π/2)/d)² · q_x^T · S · S^T · q_y
        //   S is orthogonal (randomized Hadamard): S·S^T = I
        //   = (π/2)/d² · <q_x, q_y>
        // Full: ||r_x||·||r_y|| · (π/2)/d² · sign_agreement
        // With ||r_x||·||r_y|| ≈ d · mse_per_coord:
        //   = mse_per_coord · (π/2)/d · sign_agreement
        let d = self.padded_dims as f32;
        let agreement = d - 2.0 * sign_disagree as f32;
        const PI_OVER_2: f32 = 1.5707964;
        let r_norm_product = d * self.qjl_r_norm_sq_per_coord;
        let qjl = PI_OVER_2 / (d * d) * r_norm_product * agreement;

        let corrected_dot = mse_dot + qjl;
        let inv_d = 1.0 / d;
        let cos_theta = (corrected_dot * inv_d).clamp(-1.0, 1.0);
        ni * ni + nj * nj - 2.0 * ni * nj * cos_theta
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.packed.len()  // quantized data
            + self.norms.len() * 4  // norms
            + self.signs.len() * 4  // signs
    }

    /// Get the packed bytes and norm for a single point (for QuantizedPoint)
    pub fn point_data(&self, idx: usize) -> (&[u8], f32) {
        let bpp = self.bytes_per_point();
        let offset = idx * bpp;
        (&self.packed[offset..offset + bpp], self.norms[idx])
    }

    fn bytes_per_point(&self) -> usize {
        match self.bits {
            QuantBits::Four => self.padded_dims / 2,
            QuantBits::Eight => self.padded_dims,
        }
    }

    /// Access raw packed bytes (for GPU upload)
    pub fn packed_data(&self) -> &[u8] { &self.packed }

    /// Access norms
    pub fn norms(&self) -> &[f32] { &self.norms }

    /// Get padded dimensionality
    pub fn padded_dims(&self) -> usize { self.padded_dims }

    /// Get codebook for GPU upload: centroids + MSE constant at the end
    pub fn sorted_centroids(&self) -> Vec<f32> {
        let mut cb = self.centroids.clone();
        cb.push(self.qjl_r_norm_sq_per_coord);
        cb
    }

    /// Get MSE per coordinate (for GPU QJL correction)
    pub fn mse_per_coord(&self) -> f32 { self.qjl_r_norm_sq_per_coord }
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_walsh_hadamard() {
        let mut data = vec![1.0, 0.0, 0.0, 0.0];
        walsh_hadamard_transform(&mut data);
        // WHT of [1,0,0,0] should be [0.5, 0.5, 0.5, 0.5]
        for &v in &data {
            assert!((v - 0.5).abs() < 1e-6, "got {}", v);
        }
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let data = Array2::from_shape_vec((3, 4), vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            -1.0, 0.0, 1.0, 0.5,
        ]).unwrap();

        let qdata = QuantizedData::encode(&data, 42);
        assert_eq!(qdata.n_samples, 3);

        // Dequantized values should be somewhat close to originals
        for i in 0..3 {
            let decoded = qdata.decode(i);
            let original: Vec<f64> = (0..4).map(|j| data[[i, j]]).collect();
            let error: f64 = decoded.iter().zip(original.iter())
                .map(|(&a, &b)| (a as f64 - b).powi(2))
                .sum::<f64>()
                .sqrt();
            let norm: f64 = original.iter().map(|x| x * x).sum::<f64>().sqrt();
            // Relative error should be reasonable (< 50% for 4-bit)
            assert!(error / norm < 0.5,
                "sample {}: error={:.3}, norm={:.3}, ratio={:.3}",
                i, error, norm, error / norm);
        }
    }

    #[test]
    fn test_approx_distance() {
        let data = Array2::from_shape_vec((4, 8), vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
        ]).unwrap();

        let qdata = QuantizedData::encode(&data, 42);

        // Distance from point 0 to 1 should be small (~sqrt(2))
        let d01 = qdata.approx_dist_sq(0, 1).sqrt();
        assert!((d01 - std::f32::consts::SQRT_2).abs() < 0.5,
                "d(0,1) = {}, expected ~{}", d01, std::f32::consts::SQRT_2);

        // Distance from point 0 to 3 should be large
        let d03 = qdata.approx_dist_sq(0, 3).sqrt();
        assert!(d03 > 20.0, "d(0,3) = {}, expected > 20", d03);

        // Distance ordering should be preserved: d(0,2) < d(0,3)
        let d02 = qdata.approx_dist_sq(0, 2);
        let d03 = qdata.approx_dist_sq(0, 3);
        assert!(d02 < d03, "d(0,2)={} should be < d(0,3)={}", d02, d03);
    }

    #[test]
    fn test_memory_savings() {
        let n = 1000;
        let d = 50;
        let data = Array2::zeros((n, d));
        let qdata = QuantizedData::encode(&data, 42);

        let original_bytes = n * d * 8; // f64
        let quant_bytes = qdata.memory_bytes();

        assert!(quant_bytes < original_bytes / 4,
                "quantized={} should be < original/4={}", quant_bytes, original_bytes / 4);
    }
}
