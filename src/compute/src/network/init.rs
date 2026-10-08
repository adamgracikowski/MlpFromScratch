//! Random initialization of weight matrices.

use ndarray::Array2;
use rand::Rng;
use rand::distr::{Distribution, Uniform};
use rand_distr::Normal;

use crate::config::Init;

/// Draws a `fan_in x fan_out` weight matrix for the scheme `init`.
///
/// - Xavier (Glorot) uniform: `U[-limit, limit]`, `limit = sqrt(6 / (fan_in + fan_out))`.
/// - He (Kaiming) normal: `N(0, std^2)`, `std = sqrt(2 / fan_in)`.
/// - Uniform: `U[-limit, limit]` with the configured `limit`.
///
/// Weights are drawn row by row, so the same generator state always gives
/// the same matrix.
pub fn initialize_weights<R: Rng + ?Sized>(
    init: Init,
    fan_in: usize,
    fan_out: usize,
    rng: &mut R,
) -> Array2<f64> {
    assert!(
        fan_in > 0 && fan_out > 0,
        "layer must have at least one input and one output, got {fan_in} x {fan_out}"
    );
    match init {
        Init::Xavier => {
            let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
            sample(fan_in, fan_out, &symmetric_uniform(limit), rng)
        }
        Init::He => {
            let std = (2.0 / fan_in as f64).sqrt();
            let normal = Normal::new(0.0, std).expect("standard deviation is positive");
            sample(fan_in, fan_out, &normal, rng)
        }
        Init::Uniform { limit } => sample(fan_in, fan_out, &symmetric_uniform(limit), rng),
    }
}

/// Uniform distribution on `[-limit, limit]`.
fn symmetric_uniform(limit: f64) -> Uniform<f64> {
    Uniform::new_inclusive(-limit, limit).expect("limit is positive and finite")
}

/// Fills a `rows x columns` matrix with samples, in row-major order.
fn sample<D: Distribution<f64>, R: Rng + ?Sized>(
    rows: usize,
    columns: usize,
    distribution: &D,
    rng: &mut R,
) -> Array2<f64> {
    Array2::from_shape_simple_fn((rows, columns), || distribution.sample(rng))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::{RngStream, rng};

    fn weights(init: Init, fan_in: usize, fan_out: usize, seed: u64) -> Array2<f64> {
        initialize_weights(init, fan_in, fan_out, &mut rng(seed, RngStream::Init))
    }

    fn mean_and_std(weights: &Array2<f64>) -> (f64, f64) {
        let mean = weights.mean().unwrap();
        (mean, weights.std(0.0))
    }

    #[test]
    fn has_fan_in_x_fan_out_shape() {
        for init in [Init::Xavier, Init::He, Init::Uniform { limit: 0.5 }] {
            assert_eq!(weights(init, 3, 5, 42).dim(), (3, 5));
        }
    }

    #[test]
    fn xavier_stays_within_limit() {
        let weights = weights(Init::Xavier, 40, 20, 42);
        let limit = (6.0f64 / 60.0).sqrt();
        assert!(weights.iter().all(|weight| weight.abs() <= limit));
        // Uniform on [-limit, limit] has standard deviation limit / sqrt(3).
        let (mean, std) = mean_and_std(&weights);
        assert!(mean.abs() < 0.03);
        assert!((std - limit / 3f64.sqrt()).abs() < 0.02);
    }

    #[test]
    fn uniform_stays_within_limit() {
        let weights = weights(Init::Uniform { limit: 0.25 }, 30, 30, 42);
        assert!(weights.iter().all(|weight| weight.abs() <= 0.25));
        assert!(weights.iter().any(|weight| weight.abs() > 0.2));
    }

    #[test]
    fn he_has_expected_mean_and_std() {
        let weights = weights(Init::He, 400, 250, 42);
        let expected_std = (2.0f64 / 400.0).sqrt();
        let (mean, std) = mean_and_std(&weights);
        assert!(mean.abs() < 0.002, "mean {mean}");
        assert!((std / expected_std - 1.0).abs() < 0.02, "std {std}");
    }

    #[test]
    fn same_seed_gives_same_weights() {
        for init in [Init::Xavier, Init::He, Init::Uniform { limit: 0.5 }] {
            assert_eq!(weights(init, 4, 3, 7), weights(init, 4, 3, 7));
            assert_ne!(weights(init, 4, 3, 7), weights(init, 4, 3, 8));
        }
    }

    #[test]
    #[should_panic(expected = "at least one input")]
    fn rejects_empty_layer() {
        weights(Init::Xavier, 0, 3, 42);
    }
}
