//! Activation functions and their vector-Jacobian products.

use ndarray::{Array2, Axis, Zip};
use strum::Display;

use crate::config::{HiddenActivation, OutputActivation};

/// Activation function of one layer, applied to the pre-activations
/// `Z` (`n x k`, one row per sample).
///
/// Hidden and output activations from the config map onto this one type, so
/// every layer is handled the same way.
#[derive(Debug, Clone, Copy, PartialEq, Display)]
#[strum(serialize_all = "snake_case")]
pub enum Activation {
    Identity,
    Sigmoid,
    Tanh,
    Relu,
    /// ReLU with slope `alpha` for negative inputs.
    #[strum(to_string = "leaky_relu(alpha={alpha})")]
    LeakyRelu {
        alpha: f64,
    },
    /// Row-wise softmax: every row of the output is a probability vector.
    Softmax,
}

impl Activation {
    /// Computes the activations `A = f(Z)`.
    pub fn forward(self, z: &Array2<f64>) -> Array2<f64> {
        match self {
            Activation::Identity => z.clone(),
            Activation::Sigmoid => z.mapv(sigmoid),
            Activation::Tanh => z.mapv(f64::tanh),
            Activation::Relu => z.mapv(|value| if value > 0.0 { value } else { 0.0 }),
            Activation::LeakyRelu { alpha } => {
                z.mapv(|value| if value > 0.0 { value } else { alpha * value })
            }
            Activation::Softmax => softmax(z),
        }
    }

    /// Vector-Jacobian product: turns the gradient of the loss with respect
    /// to the activations (`grad_a = dL/dA`) into the gradient with respect
    /// to the pre-activations (`dL/dZ`).
    ///
    /// `z` and `a` are the pre-activations and activations cached by the
    /// forward pass; all three matrices are `n x k`. Element-wise activations
    /// give `f'(Z) * dA`. Softmax mixes the entries of a row, so its Jacobian
    /// is not diagonal: `dZ = A * (dA - rowsum(dA * A))`.
    pub fn backward(self, z: &Array2<f64>, a: &Array2<f64>, grad_a: &Array2<f64>) -> Array2<f64> {
        assert_eq!(z.dim(), a.dim(), "Z and A must have the same shape");
        assert_eq!(grad_a.dim(), a.dim(), "dA and A must have the same shape");
        match self {
            Activation::Identity => grad_a.clone(),
            // sigmoid'(z) = a * (1 - a)
            Activation::Sigmoid => Zip::from(a)
                .and(grad_a)
                .map_collect(|&a, &grad| a * (1.0 - a) * grad),
            // tanh'(z) = 1 - a^2
            Activation::Tanh => Zip::from(a)
                .and(grad_a)
                .map_collect(|&a, &grad| (1.0 - a * a) * grad),
            // relu'(z) = 1 for z > 0, else 0 (0 at the kink)
            Activation::Relu => Zip::from(z)
                .and(grad_a)
                .map_collect(|&z, &grad| if z > 0.0 { grad } else { 0.0 }),
            Activation::LeakyRelu { alpha } => Zip::from(z)
                .and(grad_a)
                .map_collect(|&z, &grad| if z > 0.0 { grad } else { alpha * grad }),
            Activation::Softmax => {
                // Per row: dZ_j = a_j * (dA_j - sum_k dA_k * a_k).
                let weighted = (grad_a * a).sum_axis(Axis(1)).insert_axis(Axis(1));
                a * &(grad_a - &weighted)
            }
        }
    }
}

impl From<HiddenActivation> for Activation {
    fn from(activation: HiddenActivation) -> Self {
        match activation {
            HiddenActivation::Sigmoid => Activation::Sigmoid,
            HiddenActivation::Tanh => Activation::Tanh,
            HiddenActivation::Relu => Activation::Relu,
            HiddenActivation::LeakyRelu { alpha } => Activation::LeakyRelu { alpha },
        }
    }
}

impl From<OutputActivation> for Activation {
    fn from(activation: OutputActivation) -> Self {
        match activation {
            OutputActivation::Identity => Activation::Identity,
            OutputActivation::Sigmoid => Activation::Sigmoid,
            OutputActivation::Softmax => Activation::Softmax,
        }
    }
}

/// Logistic function in a form that does not overflow for large `|z|`.
fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let exp = z.exp();
        exp / (1.0 + exp)
    }
}

/// Row-wise softmax. The row maximum is subtracted before `exp`, which does
/// not change the result but keeps `exp` from overflowing.
fn softmax(z: &Array2<f64>) -> Array2<f64> {
    let mut a = z.clone();
    for mut row in a.rows_mut() {
        let max = row.fold(f64::NEG_INFINITY, |max, &value| max.max(value));
        row.mapv_inplace(|value| (value - max).exp());
        let sum = row.sum();
        row /= sum;
    }
    a
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    const ALL: [Activation; 6] = [
        Activation::Identity,
        Activation::Sigmoid,
        Activation::Tanh,
        Activation::Relu,
        Activation::LeakyRelu { alpha: 0.1 },
        Activation::Softmax,
    ];

    /// Pre-activations away from the ReLU kink, so central differences are exact enough.
    fn sample_z() -> Array2<f64> {
        array![[-1.3, 0.7, 2.1], [0.4, -0.25, -2.6]]
    }

    /// Arbitrary upstream gradient `dL/dA`.
    fn sample_grad() -> Array2<f64> {
        array![[0.5, -1.2, 0.3], [2.0, 0.8, -0.7]]
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

    /// Central-difference gradient of `L(Z) = sum(dA * f(Z))` with respect to `Z`,
    /// which is exactly what `backward` computes analytically.
    fn numerical_backward(
        activation: Activation,
        z: &Array2<f64>,
        grad_a: &Array2<f64>,
    ) -> Array2<f64> {
        let step = 1e-6;
        let loss = |z: &Array2<f64>| (grad_a * &activation.forward(z)).sum();
        Array2::from_shape_fn(z.dim(), |index| {
            let mut plus = z.clone();
            plus[index] += step;
            let mut minus = z.clone();
            minus[index] -= step;
            (loss(&plus) - loss(&minus)) / (2.0 * step)
        })
    }

    #[test]
    fn computes_known_values() {
        let z = array![[-2.0, 0.0, 3.0]];
        let cases = [
            (Activation::Identity, array![[-2.0, 0.0, 3.0]]),
            (
                Activation::Sigmoid,
                array![[1.0 / (1.0 + 2f64.exp()), 0.5, 1.0 / (1.0 + (-3f64).exp())]],
            ),
            (Activation::Tanh, array![[(-2f64).tanh(), 0.0, 3f64.tanh()]]),
            (Activation::Relu, array![[0.0, 0.0, 3.0]]),
            (
                Activation::LeakyRelu { alpha: 0.1 },
                array![[-0.2, 0.0, 3.0]],
            ),
        ];
        for (activation, expected) in cases {
            assert_close(&activation.forward(&z), &expected, 1e-12);
        }
    }

    #[test]
    fn softmax_matches_hand_computed_values() {
        let z = array![[0.0, 2f64.ln()], [1.0, 1.0]];
        let expected = array![[1.0 / 3.0, 2.0 / 3.0], [0.5, 0.5]];
        assert_close(&Activation::Softmax.forward(&z), &expected, 1e-12);
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let a = Activation::Softmax.forward(&sample_z());
        for sum in a.sum_axis(Axis(1)) {
            assert!((sum - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn stays_finite_for_extreme_inputs() {
        let z = array![[1000.0, -1000.0, 0.0], [-1000.0, -1000.0, 1000.0]];
        for activation in ALL {
            let a = activation.forward(&z);
            assert!(a.iter().all(|value| value.is_finite()), "{activation}");
        }
        assert_close(
            &Activation::Sigmoid.forward(&z),
            &array![[1.0, 0.0, 0.5], [0.0, 0.0, 1.0]],
            1e-12,
        );
        assert_close(
            &Activation::Softmax.forward(&z),
            &array![[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            1e-12,
        );
    }

    #[test]
    fn backward_matches_numerical_gradient() {
        let z = sample_z();
        let grad_a = sample_grad();
        for activation in ALL {
            let a = activation.forward(&z);
            let analytical = activation.backward(&z, &a, &grad_a);
            let numerical = numerical_backward(activation, &z, &grad_a);
            assert_close(&analytical, &numerical, 1e-8);
        }
    }

    #[test]
    fn relu_backward_blocks_negative_inputs() {
        let z = array![[-1.0, 2.0]];
        let grad_a = array![[3.0, 4.0]];
        let relu = Activation::Relu;
        let leaky = Activation::LeakyRelu { alpha: 0.1 };
        assert_eq!(
            relu.backward(&z, &relu.forward(&z), &grad_a),
            array![[0.0, 4.0]]
        );
        assert_close(
            &leaky.backward(&z, &leaky.forward(&z), &grad_a),
            &array![[0.3, 4.0]],
            1e-12,
        );
    }

    #[test]
    #[should_panic(expected = "dA and A must have the same shape")]
    fn backward_rejects_mismatched_gradient() {
        let z = sample_z();
        let a = Activation::Tanh.forward(&z);
        Activation::Tanh.backward(&z, &a, &array![[1.0, 2.0]]);
    }

    #[test]
    fn converts_config_activations() {
        let hidden = [
            HiddenActivation::Sigmoid,
            HiddenActivation::Tanh,
            HiddenActivation::Relu,
            HiddenActivation::LeakyRelu { alpha: 0.1 },
        ]
        .map(Activation::from);
        assert_eq!(hidden, ALL[1..5]);
        let output = [
            OutputActivation::Identity,
            OutputActivation::Sigmoid,
            OutputActivation::Softmax,
        ]
        .map(Activation::from);
        assert_eq!(
            output,
            [
                Activation::Identity,
                Activation::Sigmoid,
                Activation::Softmax
            ]
        );
    }

    #[test]
    fn displays_config_names() {
        let names = ALL.map(|activation| activation.to_string());
        assert_eq!(
            names,
            [
                "identity",
                "sigmoid",
                "tanh",
                "relu",
                "leaky_relu(alpha=0.1)",
                "softmax"
            ]
        );
    }

    #[test]
    fn display_matches_config_display() {
        let hidden = HiddenActivation::LeakyRelu { alpha: 0.01 };
        assert_eq!(Activation::from(hidden).to_string(), hidden.to_string());
        let output = OutputActivation::Softmax;
        assert_eq!(Activation::from(output).to_string(), output.to_string());
    }
}
