use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::Rng;

/// TurboQuant-inspired vector quantization for fast approximate distance computation.
///
/// Algorithm:
/// 1. Normalize vectors to unit length (store norms separately)
/// 2. Apply randomized Hadamard transform (random sign flips + Walsh-Hadamard)
/// 3. Scalar quantize each coordinate to b bits using Lloyd-Max codebook
/// 4. For distance computation: dequantize to f32
///
/// This preserves distances with near-optimal distortion guarantees.

/// Lloyd-Max codebook for 4-bit MSE quantization (16 levels) of N(0,1)
const LLOYD_MAX_4BIT_CENTROIDS: [f32; 16] = [
    -2.4008, -1.7479, -1.2461, -0.9224, -0.6568, -0.4246, -0.2127, -0.0638,
     0.0638,  0.2127,  0.4246,  0.6568,  0.9224,  1.2461,  1.7479,  2.4008,
];

/// Lloyd-Max codebook for 3-bit MSE quantization (8 levels) of N(0,1)
/// Used in TurboQuant_prod where 1 bit goes to QJL sign
const LLOYD_MAX_3BIT_CENTROIDS: [f32; 8] = [
    -1.7479, -1.0500, -0.5006, -0.0638,
     0.0638,  0.5006,  1.0500,  1.7479,
];

fn sorted_codebook() -> ([f32; 16], [f32; 15]) {
    let centroids = LLOYD_MAX_4BIT_CENTROIDS;
    let mut boundaries = [0.0f32; 15];
    for i in 0..15 {
        boundaries[i] = (centroids[i] + centroids[i + 1]) / 2.0;
    }
    (centroids, boundaries)
}

fn sorted_codebook_3bit() -> ([f32; 8], [f32; 7]) {
    let centroids = LLOYD_MAX_3BIT_CENTROIDS;
    let mut boundaries = [0.0f32; 7];
    for i in 0..7 {
        boundaries[i] = (centroids[i] + centroids[i + 1]) / 2.0;
    }
    (centroids, boundaries)
}

/// Quantize a scalar value to 4-bit index (16 levels)
#[inline]
fn quantize_scalar(val: f32, boundaries: &[f32; 15]) -> u8 {
    let mut idx = 0u8;
    for (i, &b) in boundaries.iter().enumerate() {
        if val > b { idx = (i + 1) as u8; } else { break; }
    }
    idx
}

/// Quantize a scalar value to 3-bit index (8 levels)
#[inline]
fn quantize_scalar_3bit(val: f32, boundaries: &[f32; 7]) -> u8 {
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
/// Packing (4-bit mode): nibble = (3-bit MSE index) << 1 | sign_bit
/// Packing (8-bit mode): byte  = (7-bit MSE index) << 1 | sign_bit
///
/// For inner product: dot ≈ mse_dot + √(π/2)/d · ||r_i|| · ||r_j|| · sign_agreement
/// sign_agreement = d - 2·hamming(signs_i, signs_j), computed via XOR+popcount.
pub struct QuantizedData {
    pub n_samples: usize,
    pub n_dims: usize,
    padded_dims: usize,
    bits: QuantBits,
    /// Random sign flips for Hadamard randomization
    signs: Vec<f32>,
    /// Norms of original vectors
    norms: Vec<f32>,
    /// Packed data: each coordinate has (b-1) MSE bits + 1 QJL sign bit
    packed: Vec<u8>,
    /// MSE codebook centroids (3-bit for Four, 7-bit uniform for Eight)
    centroids_3bit: [f32; 8],
    /// 8-bit: range for 7-bit uniform quantizer
    quant7_range: f32,
    /// Residual norms ||r_i|| per point (for QJL correction)
    residual_norms: Vec<f32>,
}

/// Range for 8-bit uniform quantizer (covers ~99.7% of N(0,1))
const QUANT8_RANGE: f32 = 3.5;

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

        let (centroids_3bit, boundaries_3bit) = sorted_codebook_3bit();

        let mut rng = StdRng::seed_from_u64(seed);
        let signs: Vec<f32> = (0..padded_dims)
            .map(|_| { let v: f32 = rng.gen_range(0.0..1.0); if v < 0.5 { 1.0 } else { -1.0 } })
            .collect();

        let mut norms = Vec::with_capacity(n_samples);
        let mut residual_norms = Vec::with_capacity(n_samples);
        let bytes_per_point = match bits {
            QuantBits::Four => padded_dims / 2,
            QuantBits::Eight => padded_dims,
        };
        let mut packed = Vec::with_capacity(n_samples * bytes_per_point);

        for i in 0..n_samples {
            let mut vec = vec![0.0f32; padded_dims];
            for j in 0..n_dims {
                vec[j] = data[[i, j]] as f32;
            }

            let norm: f32 = vec.iter().map(|&v| v * v).sum::<f32>().sqrt();
            norms.push(norm);

            if norm > f32::EPSILON {
                for v in vec.iter_mut() { *v /= norm; }
            }

            for (v, &s) in vec.iter_mut().zip(signs.iter()) { *v *= s; }
            walsh_hadamard_transform(&mut vec);
            let scale = (padded_dims as f32).sqrt();
            for v in vec.iter_mut() { *v *= scale; }

            // Quantize + compute residual sign (QJL) in one pass
            // Pack: (b-1)-bit MSE index shifted left by 1, OR'd with sign(residual)
            let mut r_norm_sq = 0.0f32;

            match bits {
                QuantBits::Four => {
                    // 3-bit MSE (8 levels) + 1-bit sign = 4 bits per coordinate
                    // Pack two coordinates per byte: hi nibble + lo nibble
                    for j in (0..padded_dims).step_by(2) {
                        let idx_hi = quantize_scalar_3bit(vec[j], &boundaries_3bit);
                        let residual_hi = vec[j] - centroids_3bit[idx_hi as usize];
                        let sign_hi: u8 = if residual_hi >= 0.0 { 1 } else { 0 };
                        r_norm_sq += residual_hi * residual_hi;

                        let idx_lo = quantize_scalar_3bit(vec[j + 1], &boundaries_3bit);
                        let residual_lo = vec[j + 1] - centroids_3bit[idx_lo as usize];
                        let sign_lo: u8 = if residual_lo >= 0.0 { 1 } else { 0 };
                        r_norm_sq += residual_lo * residual_lo;

                        // nibble = (3-bit idx << 1) | sign
                        let hi = (idx_hi << 1) | sign_hi;
                        let lo = (idx_lo << 1) | sign_lo;
                        packed.push((hi << 4) | lo);
                    }
                }
                QuantBits::Eight => {
                    // 7-bit MSE (128 levels, uniform) + 1-bit sign = 8 bits per coordinate
                    let range = QUANT8_RANGE;
                    for j in 0..padded_dims {
                        let clamped = vec[j].clamp(-range, range);
                        let idx = ((clamped + range) / (2.0 * range) * 127.0) as u8; // 0..127
                        let dequant = (idx as f32 / 127.0) * 2.0 * range - range;
                        let residual = vec[j] - dequant;
                        let sign: u8 = if residual >= 0.0 { 1 } else { 0 };
                        r_norm_sq += residual * residual;

                        // byte = (7-bit idx << 1) | sign
                        packed.push((idx << 1) | sign);
                    }
                }
            }

            residual_norms.push(r_norm_sq.sqrt());
        }

        Self {
            n_samples,
            n_dims,
            padded_dims,
            bits,
            signs,
            norms,
            packed,
            centroids_3bit,
            quant7_range: QUANT8_RANGE,
            residual_norms,
        }
    }


    /// Dequantize a single vector to f32 (MSE part only, ignoring QJL sign)
    pub fn decode(&self, idx: usize) -> Vec<f32> {
        let bpp = self.bytes_per_point();
        let offset = idx * bpp;

        let mut vec = vec![0.0f32; self.padded_dims];
        match self.bits {
            QuantBits::Four => {
                // nibble = (3-bit idx << 1) | sign_bit
                for j in 0..self.padded_dims / 2 {
                    let byte = self.packed[offset + j];
                    let hi_idx = ((byte >> 4) & 0x0F) >> 1; // top 3 bits of hi nibble
                    let lo_idx = (byte & 0x0F) >> 1;         // top 3 bits of lo nibble
                    vec[j * 2] = self.centroids_3bit[hi_idx as usize];
                    vec[j * 2 + 1] = self.centroids_3bit[lo_idx as usize];
                }
            }
            QuantBits::Eight => {
                // byte = (7-bit idx << 1) | sign_bit
                let range = self.quant7_range;
                for j in 0..self.padded_dims {
                    let idx = self.packed[offset + j] >> 1; // top 7 bits
                    vec[j] = (idx as f32 / 127.0) * 2.0 * range - range;
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
                    dot += self.centroids_3bit[idx_i_hi as usize] * self.centroids_3bit[idx_j_hi as usize];
                    disagree += (sign_i_hi ^ sign_j_hi) as u32;
                    // Lo nibble
                    let idx_i_lo = (bi & 0x0F) >> 1;
                    let idx_j_lo = (bj & 0x0F) >> 1;
                    let sign_i_lo = bi & 1;
                    let sign_j_lo = bj & 1;
                    dot += self.centroids_3bit[idx_i_lo as usize] * self.centroids_3bit[idx_j_lo as usize];
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
                let range = self.quant7_range;
                for k in 0..bpp {
                    let bi = self.packed[off_i + k];
                    let bj = self.packed[off_j + k];
                    // byte = (7-bit idx << 1) | sign
                    let idx_i = bi >> 1;
                    let idx_j = bj >> 1;
                    let sign_i = bi & 1;
                    let sign_j = bj & 1;
                    let vi = (idx_i as f32 / 127.0) * 2.0 * range - range;
                    let vj = (idx_j as f32 / 127.0) * 2.0 * range - range;
                    dot += vi * vj;
                    disagree += (sign_i ^ sign_j) as u32;
                }
                (dot, disagree)
            }
        };

        // QJL correction: sqrt(π/2)/d · ||r_i|| · ||r_j|| · (d - 2·hamming)
        let d = self.padded_dims as f32;
        let agreement = d - 2.0 * sign_disagree as f32;
        const SQRT_PI_OVER_2: f32 = 1.2533141;
        let qjl = SQRT_PI_OVER_2 / d * self.residual_norms[i] * self.residual_norms[j] * agreement;

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

    /// Get sorted codebook centroids (for GPU upload)
    pub fn sorted_centroids(&self) -> Vec<f32> {
        // GPU kernel needs the 3-bit codebook for TQ4 mode
        self.centroids_3bit.to_vec()
    }
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
