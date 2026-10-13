//! Loss functions: the mean loss of a batch and its gradient.

use ndarray::{Array2, Zip};

use super::Activation;
use crate::config::Loss;

impl Loss {
    /// Mean loss of a batch from the output layer's `z`, `a` and `targets`
    /// (all `n x K`). MSE and MAE average over all entries; cross-entropy
    /// averages over samples and uses the log-softmax of `z` for stability.
    pub fn value(self, z: &Array2<f64>, a: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        assert_shapes(z, a, targets);
        match self {
            Loss::CrossEntropy => {
                let total: f64 = z
                    .rows()
                    .into_iter()
                    .zip(targets.rows())
                    .map(|(z, y)| {
                        // logsumexp with the row maximum taken out, so exp cannot overflow.
                        let max = z.fold(f64::NEG_INFINITY, |max, &value| max.max(value));
                        let log_sum_exp = max + z.mapv(|value| (value - max).exp()).sum().ln();
                        // -sum_k y_k (z_k - logsumexp(z))
                        y.sum() * log_sum_exp - y.dot(&z)
                    })
                    .sum();
                total / a.nrows() as f64
            }
            Loss::Mse => mean(
                Zip::from(a)
                    .and(targets)
                    .map_collect(|&a, &y| (a - y).powi(2)),
            ),
            Loss::Mae => mean(
                Zip::from(a)
                    .and(targets)
                    .map_collect(|&a, &y| (a - y).abs()),
            ),
        }
    }

    /// Gradient `dL/dZ` (`n x K`) of the output layer, where backpropagation
    /// starts. Cross-entropy with softmax uses the shortcut `(A - Y) / n`;
    /// other losses pass `dL/dA` through `activation.backward`.
    pub fn output_gradient(
        self,
        activation: Activation,
        z: &Array2<f64>,
        a: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Array2<f64> {
        assert_shapes(z, a, targets);
        match (self, activation) {
            (Loss::CrossEntropy, Activation::Softmax) => (a - targets) / a.nrows() as f64,
            (Loss::CrossEntropy, _) => {
                panic!("cross_entropy requires a softmax output, not {activation}")
            }
            _ => activation.backward(z, a, &self.activation_gradient(a, targets)),
        }
    }

    /// Gradient `dL/dA` (`n x K`). For cross-entropy, `-Y / (n A)` is unstable
    /// near `A = 0`, hence the shortcut in `output_gradient`.
    fn activation_gradient(self, a: &Array2<f64>, targets: &Array2<f64>) -> Array2<f64> {
        let samples = a.nrows() as f64;
        let entries = a.len() as f64;
        let zip = Zip::from(a).and(targets);
        match self {
            Loss::CrossEntropy => zip.map_collect(|&a, &y| -y / (samples * a)),
            // d/da (a - y)^2 = 2 (a - y)
            Loss::Mse => zip.map_collect(|&a, &y| 2.0 * (a - y) / entries),
            // d/da |a - y| = sign(a - y), 0 at the kink
            Loss::Mae => zip.map_collect(|&a, &y| sign(a - y) / entries),
        }
    }
}

/// Checks that the outputs and targets describe the same non-empty batch.
fn assert_shapes(z: &Array2<f64>, a: &Array2<f64>, targets: &Array2<f64>) {
    assert_eq!(z.dim(), a.dim(), "Z and A must have the same shape");
    assert_eq!(
        targets.dim(),
        a.dim(),
        "targets must have the shape of the outputs"
    );
    assert!(a.nrows() > 0, "a batch needs at least one sample");
}

/// Mean of all entries of a non-empty matrix.
fn mean(values: Array2<f64>) -> f64 {
    values.mean().expect("a batch has at least one entry")
}

/// Sign of `value`, with 0 for 0 (unlike `f64::signum`, which gives 1).
fn sign(value: f64) -> f64 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        -1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    /// Every (loss, output activation) pair allowed by config validation.
    const PAIRS: [(Loss, Activation); 5] = [
        (Loss::CrossEntropy, Activation::Softmax),
        (Loss::Mse, Activation::Softmax),
        (Loss::Mse, Activation::Sigmoid),
        (Loss::Mse, Activation::Identity),
        (Loss::Mae, Activation::Identity),
    ];

    /// Pre-activations whose outputs stay away from the MAE kink `a = y`.
    fn sample_z() -> Array2<f64> {
        array![[-1.3, 0.7, 2.1], [0.4, -0.25, -2.6]]
    }

    /// One-hot targets for `sample_z`.
    fn one_hot() -> Array2<f64> {
        array![[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    }

    fn assert_close(actual: &Array2<f64>, expected: &Array2<f64>, tolerance: f64) {
        assert_eq!(actual.dim(), expected.dim());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= tolerance,
                "{actual} differs from {expected}"
            );
        }
    }

    /// Central-difference gradient of `L(f(Z))` with respect to `Z`.
    fn numerical_gradient(
        loss: Loss,
        activation: Activation,
        z: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Array2<f64> {
        let step = 1e-6;
        let value = |z: &Array2<f64>| loss.value(z, &activation.forward(z), targets);
        Array2::from_shape_fn(z.dim(), |index| {
            let mut plus = z.clone();
            plus[index] += step;
            let mut minus = z.clone();
            minus[index] -= step;
            (value(&plus) - value(&minus)) / (2.0 * step)
        })
    }

    #[test]
    fn mse_and_mae_average_over_all_entries() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let targets = array![[0.0, 2.0], [5.0, 4.0]];
        // Differences 1, 0, -2, 0.
        assert_eq!(Loss::Mse.value(&a, &a, &targets), (1.0 + 4.0) / 4.0);
        assert_eq!(Loss::Mae.value(&a, &a, &targets), (1.0 + 2.0) / 4.0);
    }

    #[test]
    fn cross_entropy_matches_hand_computed_value() {
        // softmax([0, ln 2]) = [1/3, 2/3]; the target is the second class.
        let z = array![[0.0, 2f64.ln()], [1.0, 1.0]];
        let a = Activation::Softmax.forward(&z);
        let targets = array![[0.0, 1.0], [1.0, 0.0]];
        let expected = (-(2.0 / 3.0f64).ln() - 0.5f64.ln()) / 2.0;
        let value = Loss::CrossEntropy.value(&z, &a, &targets);
        assert!((value - expected).abs() < 1e-12, "{value}");
    }

    #[test]
    fn cross_entropy_equals_mean_negative_log_likelihood() {
        let z = sample_z();
        let a = Activation::Softmax.forward(&z);
        let targets = one_hot();
        let expected = -(&targets * &a.ln()).sum() / 2.0;
        let value = Loss::CrossEntropy.value(&z, &a, &targets);
        assert!((value - expected).abs() < 1e-12, "{value}");
    }

    #[test]
    fn cross_entropy_stays_finite_for_extreme_inputs() {
        let z = array![[1000.0, -1000.0], [1000.0, -1000.0]];
        let a = Activation::Softmax.forward(&z);
        // Right class with certainty, then wrong class with certainty.
        let targets = array![[1.0, 0.0], [0.0, 1.0]];
        let value = Loss::CrossEntropy.value(&z, &a, &targets);
        assert!((value - 2000.0 / 2.0).abs() < 1e-9, "{value}");
    }

    #[test]
    fn output_gradient_matches_numerical_gradient() {
        let z = sample_z();
        for (loss, activation) in PAIRS {
            let targets = match activation {
                Activation::Identity => array![[0.5, -1.0, 3.0], [-0.2, 1.5, 0.0]],
                _ => one_hot(),
            };
            let a = activation.forward(&z);
            let analytical = loss.output_gradient(activation, &z, &a, &targets);
            let numerical = numerical_gradient(loss, activation, &z, &targets);
            assert_close(&analytical, &numerical, 1e-8);
        }
    }

    #[test]
    fn cross_entropy_shortcut_equals_general_path() {
        let z = sample_z();
        let a = Activation::Softmax.forward(&z);
        let targets = one_hot();
        let shortcut = Loss::CrossEntropy.output_gradient(Activation::Softmax, &z, &a, &targets);
        let grad_a = Loss::CrossEntropy.activation_gradient(&a, &targets);
        let general = Activation::Softmax.backward(&z, &a, &grad_a);
        assert_close(&shortcut, &general, 1e-12);
    }

    #[test]
    fn mae_gradient_is_zero_at_the_kink() {
        let a = array![[1.0, 2.0]];
        let targets = array![[1.0, 0.0]];
        let gradient = Loss::Mae.output_gradient(Activation::Identity, &a, &a, &targets);
        assert_eq!(gradient, array![[0.0, 0.5]]);
    }

    #[test]
    #[should_panic(expected = "cross_entropy requires a softmax output, not sigmoid")]
    fn cross_entropy_rejects_non_softmax_output() {
        let z = sample_z();
        let a = Activation::Sigmoid.forward(&z);
        Loss::CrossEntropy.output_gradient(Activation::Sigmoid, &z, &a, &one_hot());
    }

    #[test]
    #[should_panic(expected = "targets must have the shape of the outputs")]
    fn rejects_mismatched_targets() {
        let z = sample_z();
        Loss::Mse.value(&z, &z, &array![[1.0, 2.0, 3.0]]);
    }
}
