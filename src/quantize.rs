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

/// Precomputed Lloyd-Max codebook for 4-bit quantization (16 levels)
/// of coordinates following a Beta((d-1)/2, (d-1)/2) distribution,
/// which for large d approximates N(0, 1/d).
/// These are the centroids for the standard normal distribution, scaled.
/// 4-bit Lloyd-Max centroids for N(0,1), from classical quantization theory.
const LLOYD_MAX_4BIT_CENTROIDS: [f32; 16] = [
    -1.7479, -1.2461, -0.9224, -0.6568,
    -0.4246, -0.2127, -0.0638,  0.0638,
     0.2127,  0.4246,  0.6568,  0.9224,
     1.2461,  1.7479,  2.4008, -2.4008,
];

/// Sorted centroids and decision boundaries for encoding
fn sorted_codebook() -> ([f32; 16], [f32; 15]) {
    let mut centroids = LLOYD_MAX_4BIT_CENTROIDS;
    centroids.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Decision boundaries are midpoints between adjacent centroids
    let mut boundaries = [0.0f32; 15];
    for i in 0..15 {
        boundaries[i] = (centroids[i] + centroids[i + 1]) / 2.0;
    }
    (centroids, boundaries)
}

/// Quantize a scalar value to 4-bit index using the codebook
#[inline]
fn quantize_scalar(val: f32, boundaries: &[f32; 15]) -> u8 {
    // Binary search for the right bucket
    let mut idx = 0u8;
    for (i, &b) in boundaries.iter().enumerate() {
        if val > b {
            idx = (i + 1) as u8;
        } else {
            break;
        }
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

/// Quantized vector storage
pub struct QuantizedData {
    /// Number of original samples
    pub n_samples: usize,
    /// Original dimensionality
    pub n_dims: usize,
    /// Padded dimensionality (power of 2)
    padded_dims: usize,
    /// Bit-width used
    bits: QuantBits,
    /// Random sign flips for Hadamard randomization
    signs: Vec<f32>,
    /// Norms of original vectors (for distance reconstruction)
    norms: Vec<f32>,
    /// Quantized data storage
    /// 4-bit: packed 2 values per byte, [n_samples][padded_dims/2]
    /// 8-bit: one value per byte, [n_samples][padded_dims]
    packed: Vec<u8>,
    /// Precomputed 4-bit codebook (unused in 8-bit mode)
    centroids_4bit: [f32; 16],
    /// 8-bit uniform quantizer range: values mapped to [-QUANT8_RANGE, +QUANT8_RANGE]
    quant8_range: f32,
}

/// Range for 8-bit uniform quantizer (covers ~99.7% of N(0,1))
const QUANT8_RANGE: f32 = 3.5;

impl QuantizedData {
    /// Quantize with 4-bit (default)
    pub fn encode(data: &Array2<f64>, seed: u64) -> Self {
        Self::encode_with_bits(data, seed, QuantBits::Four)
    }

    /// Quantize with specified bit-width
    pub fn encode_with_bits(data: &Array2<f64>, seed: u64, bits: QuantBits) -> Self {
        let n_samples = data.nrows();
        let n_dims = data.ncols();
        let padded_dims = n_dims.next_power_of_two();

        let (centroids_4bit, boundaries_4bit) = sorted_codebook();

        let mut rng = StdRng::seed_from_u64(seed);
        let signs: Vec<f32> = (0..padded_dims)
            .map(|_| { let v: f32 = rng.gen_range(0.0..1.0); if v < 0.5 { 1.0 } else { -1.0 } })
            .collect();

        let mut norms = Vec::with_capacity(n_samples);
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
                for v in vec.iter_mut() {
                    *v /= norm;
                }
            }

            for (v, &s) in vec.iter_mut().zip(signs.iter()) {
                *v *= s;
            }

            walsh_hadamard_transform(&mut vec);

            let scale = (padded_dims as f32).sqrt();
            for v in vec.iter_mut() {
                *v *= scale;
            }

            match bits {
                QuantBits::Four => {
                    for j in (0..padded_dims).step_by(2) {
                        let hi = quantize_scalar(vec[j], &boundaries_4bit);
                        let lo = quantize_scalar(vec[j + 1], &boundaries_4bit);
                        packed.push((hi << 4) | lo);
                    }
                }
                QuantBits::Eight => {
                    for j in 0..padded_dims {
                        // Uniform quantizer: map [-RANGE, +RANGE] to [0, 255]
                        let clamped = vec[j].clamp(-QUANT8_RANGE, QUANT8_RANGE);
                        let normalized = (clamped + QUANT8_RANGE) / (2.0 * QUANT8_RANGE);
                        packed.push((normalized * 255.0) as u8);
                    }
                }
            }
        }

        Self {
            n_samples,
            n_dims,
            padded_dims,
            bits,
            signs,
            norms,
            packed,
            centroids_4bit,
            quant8_range: QUANT8_RANGE,
        }
    }

    /// Dequantize a single vector to f32
    pub fn decode(&self, idx: usize) -> Vec<f32> {
        let bpp = self.bytes_per_point();
        let offset = idx * bpp;

        let mut vec = vec![0.0f32; self.padded_dims];
        match self.bits {
            QuantBits::Four => {
                for j in 0..self.padded_dims / 2 {
                    let byte = self.packed[offset + j];
                    let hi = (byte >> 4) & 0x0F;
                    let lo = byte & 0x0F;
                    vec[j * 2] = dequantize_scalar(hi, &self.centroids_4bit);
                    vec[j * 2 + 1] = dequantize_scalar(lo, &self.centroids_4bit);
                }
            }
            QuantBits::Eight => {
                let scale = 2.0 * self.quant8_range / 255.0;
                for j in 0..self.padded_dims {
                    vec[j] = self.packed[offset + j] as f32 * scale - self.quant8_range;
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
    pub fn approx_dist_sq(&self, i: usize, j: usize) -> f32 {
        let ni = self.norms[i];
        let nj = self.norms[j];

        let dot = match self.bits {
            QuantBits::Four => {
                let bpp = self.padded_dims / 2;
                let offset_i = i * bpp;
                let offset_j = j * bpp;
                let mut dot = 0.0f32;
                for k in 0..bpp {
                    let bi = self.packed[offset_i + k];
                    let bj = self.packed[offset_j + k];
                    let vi_0 = dequantize_scalar((bi >> 4) & 0x0F, &self.centroids_4bit);
                    let vi_1 = dequantize_scalar(bi & 0x0F, &self.centroids_4bit);
                    let vj_0 = dequantize_scalar((bj >> 4) & 0x0F, &self.centroids_4bit);
                    let vj_1 = dequantize_scalar(bj & 0x0F, &self.centroids_4bit);
                    dot += vi_0 * vj_0 + vi_1 * vj_1;
                }
                dot
            }
            QuantBits::Eight => {
                let bpp = self.padded_dims;
                let offset_i = i * bpp;
                let offset_j = j * bpp;
                let mut dot = 0.0f32;
                let scale = 2.0 * self.quant8_range / 255.0;
                for k in 0..bpp {
                    let vi = self.packed[offset_i + k] as f32 * scale - self.quant8_range;
                    let vj = self.packed[offset_j + k] as f32 * scale - self.quant8_range;
                    dot += vi * vj;
                }
                dot
            }
        };

        let inv_d = 1.0 / self.padded_dims as f32;
        let cos_theta = (dot * inv_d).clamp(-1.0, 1.0);
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
