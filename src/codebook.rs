/// Lloyd-Max optimal scalar quantizer for the Beta distribution
/// arising from random rotation of d-dimensional unit vectors.
///
/// After rotation, each coordinate follows:
///   f(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1-x²)^((d-3)/2)  for x ∈ [-1, 1]
///
/// For large d this converges to N(0, 1/d).
/// We solve Lloyd-Max exactly for the true Beta(d) distribution using
/// Gauss-Legendre quadrature.

use gauss_quad::GaussLegendre;

/// PDF of a single coordinate after random rotation of a d-dim unit vector.
/// This is a symmetric Beta distribution on [-1, 1].
fn beta_pdf(x: f64, d: usize) -> f64 {
    if x.abs() >= 1.0 { return 0.0; }
    let half_d = d as f64 / 2.0;
    let half_dm1 = (d as f64 - 1.0) / 2.0;
    // Coefficient: Γ(d/2) / (√π · Γ((d-1)/2))
    // Use ln_gamma for numerical stability
    let log_coeff = ln_gamma(half_d) - 0.5 * std::f64::consts::PI.ln() - ln_gamma(half_dm1);
    let log_body = ((d as f64 - 3.0) / 2.0) * (1.0 - x * x).ln();
    (log_coeff + log_body).exp()
}

/// Log-gamma function via Stirling's approximation + Lanczos for small values
fn ln_gamma(x: f64) -> f64 {
    // Use the standard library's gamma via exp/ln trick
    // For positive x, Γ(x) = (x-1)!  for integers
    // Use Lanczos approximation
    if x <= 0.0 { return f64::INFINITY; }
    if x < 0.5 {
        // Reflection formula: Γ(x)·Γ(1-x) = π/sin(πx)
        let pi = std::f64::consts::PI;
        return pi.ln() - (pi * x).sin().ln() - ln_gamma(1.0 - x);
    }
    // Lanczos coefficients (g=7)
    let g = 7.0f64;
    let c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    let z = x - 1.0;
    let mut sum = c[0];
    for i in 1..9 {
        sum += c[i] / (z + i as f64);
    }
    let t = z + g + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (t.ln() * (z + 0.5)) - t + sum.ln()
}

/// Integrate f(x) over [a, b] using Gauss-Legendre quadrature
fn integrate<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, quad: &GaussLegendre) -> f64 {
    quad.integrate(a, b, f)
}

/// Solve Lloyd-Max optimal quantizer for d-dimensional Beta distribution.
/// Returns (centroids, boundaries) for the given number of levels.
pub fn solve_lloyd_max(d: usize, n_levels: usize) -> (Vec<f32>, Vec<f32>) {
    let sigma = 1.0 / (d as f64).sqrt();
    let lo = -3.5 * sigma;
    let hi = 3.5 * sigma;

    // Initialize centroids uniformly
    let mut centroids: Vec<f64> = (0..n_levels)
        .map(|i| lo + (hi - lo) * (i as f64 + 0.5) / n_levels as f64)
        .collect();

    // Gauss-Legendre quadrature with 32 points (more than enough for smooth PDFs)
    let quad = GaussLegendre::new(32).unwrap();

    let pdf = |x: f64| beta_pdf(x, d);

    for _iter in 0..200 {
        // Compute boundaries (midpoints)
        let mut boundaries = Vec::with_capacity(n_levels - 1);
        for i in 0..n_levels - 1 {
            boundaries.push((centroids[i] + centroids[i + 1]) / 2.0);
        }

        // Update centroids as conditional expectations
        let mut edges = vec![lo * 3.0];
        edges.extend_from_slice(&boundaries);
        edges.push(hi * 3.0);

        let mut new_centroids = Vec::with_capacity(n_levels);
        let mut max_shift = 0.0f64;

        for i in 0..n_levels {
            let a = edges[i];
            let b = edges[i + 1];

            let numer = integrate(&|x| x * pdf(x), a, b, &quad);
            let denom = integrate(&pdf, a, b, &quad);

            let c = if denom > 1e-15 { numer / denom } else { centroids[i] };
            max_shift = max_shift.max((c - centroids[i]).abs());
            new_centroids.push(c);
        }

        centroids = new_centroids;

        if max_shift < 1e-12 {
            break;
        }
    }

    // Final boundaries
    let boundaries: Vec<f32> = (0..n_levels - 1)
        .map(|i| ((centroids[i] + centroids[i + 1]) / 2.0) as f32)
        .collect();
    let centroids: Vec<f32> = centroids.iter().map(|&c| c as f32).collect();

    (centroids, boundaries)
}

/// Compute expected MSE distortion per coordinate
pub fn compute_distortion(d: usize, centroids: &[f32], boundaries: &[f32]) -> f32 {
    let quad = GaussLegendre::new(32).unwrap();
    let pdf = |x: f64| beta_pdf(x, d);
    let sigma = 1.0 / (d as f64).sqrt();
    let lo = -3.5 * sigma * 3.0;
    let hi = 3.5 * sigma * 3.0;

    let mut edges = vec![lo];
    for &b in boundaries { edges.push(b as f64); }
    edges.push(hi);

    let mut total = 0.0f64;
    for i in 0..centroids.len() {
        let c = centroids[i] as f64;
        let a = edges[i];
        let b = edges[i + 1];
        total += integrate(|x: f64| (x - c).powi(2) * pdf(x), a, b, &quad);
    }
    total as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beta_pdf_integrates_to_one() {
        let quad = GaussLegendre::new(64).unwrap();
        for &d in &[8, 32, 64, 128] {
            let total = integrate(&|x| beta_pdf(x, d), -1.0, 1.0, &quad);
            assert!((total - 1.0).abs() < 0.01,
                    "d={}: integral={}, expected 1.0", d, total);
        }
    }

    #[test]
    fn test_lloyd_max_8_levels() {
        let (centroids, boundaries) = solve_lloyd_max(32, 8);
        assert_eq!(centroids.len(), 8);
        assert_eq!(boundaries.len(), 7);
        // Centroids should be sorted
        for i in 1..8 {
            assert!(centroids[i] > centroids[i - 1]);
        }
        // Should be symmetric around 0
        assert!((centroids[0] + centroids[7]).abs() < 0.01);
    }

    #[test]
    fn test_lloyd_max_varies_with_d() {
        let (c32, _) = solve_lloyd_max(32, 8);
        let (c128, _) = solve_lloyd_max(128, 8);
        // Higher d → narrower distribution → centroids closer to 0
        assert!(c128[7].abs() < c32[7].abs(),
                "d=128 centroids should be tighter: {} vs {}", c128[7], c32[7]);
    }

    #[test]
    fn test_distortion_decreases_with_bits() {
        let (c8, b8) = solve_lloyd_max(32, 8);     // 3-bit
        let (c128, b128) = solve_lloyd_max(32, 128); // 7-bit
        let d8 = compute_distortion(32, &c8, &b8);
        let d128 = compute_distortion(32, &c128, &b128);
        assert!(d128 < d8, "7-bit distortion {} should be less than 3-bit {}", d128, d8);
    }
}
