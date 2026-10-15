//! Optimizers: update rules that turn gradients into weight changes.
//!
//! Sources of the update rules:
//! - momentum: Sutskever et al. (2013), "On the importance of initialization
//!   and momentum in deep learning", equation (1) (classical momentum of
//!   Polyak, 1964); Goodfellow et al., "Deep Learning", algorithm 8.2,
//! - Adam: Kingma and Ba (2015), "Adam: A Method for Stochastic
//!   Optimization", algorithm 1.

use ndarray::{Array1, Zip};

use crate::config::Optimizer;
use crate::network::{Gradients, Mlp};

/// An optimizer with its hyperparameters and the state it keeps between
/// steps. Every update is applied to all weights and biases element-wise.
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizerState {
    /// Plain SGD: `w = w - lr * g`.
    Sgd { learning_rate: f64 },
    /// Classical momentum: `v = momentum * v - lr * g`, `w = w + v`.
    Momentum {
        learning_rate: f64,
        momentum: f64,
        /// `v`: the last weight change, shaped like the parameters.
        velocity: Gradients,
    },
    /// Adam with bias-corrected moment estimates.
    Adam {
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        /// `m`: moving average of the gradients.
        first_moment: Gradients,
        /// `v`: moving average of the squared gradients.
        second_moment: Gradients,
        /// `t`: number of steps taken so far.
        step: i32,
    },
}

impl OptimizerState {
    /// Creates the `optimizer` for `network`, with buffers (velocity or
    /// moments) set to zero.
    pub fn new(optimizer: Optimizer, network: &Mlp) -> Self {
        match optimizer {
            Optimizer::Sgd { learning_rate } => Self::Sgd { learning_rate },
            Optimizer::Momentum {
                learning_rate,
                momentum,
            } => Self::Momentum {
                learning_rate,
                momentum,
                velocity: Gradients::zeros_like(network),
            },
            Optimizer::Adam {
                learning_rate,
                beta1,
                beta2,
                epsilon,
            } => Self::Adam {
                learning_rate,
                beta1,
                beta2,
                epsilon,
                first_moment: Gradients::zeros_like(network),
                second_moment: Gradients::zeros_like(network),
                step: 0,
            },
        }
    }

    /// Updates the weights and biases of `network` in place from the
    /// `gradients` of the loss (one batch), and the optimizer state.
    ///
    /// Panics when the gradients are not shaped like the parameters.
    pub fn step(&mut self, network: &mut Mlp, gradients: &Gradients) {
        assert_shapes_match(network, gradients);
        // Every parameter tensor (weights, then bias, layer by layer) paired
        // with its gradient; buffers are zipped in the same order.
        let parameters = network.parameters_mut().zip(gradients.tensors());
        match self {
            Self::Sgd { learning_rate } => {
                let learning_rate = *learning_rate;
                for (mut parameter, gradient) in parameters {
                    Zip::from(&mut parameter)
                        .and(&gradient)
                        .for_each(|w, &g| *w -= learning_rate * g);
                }
            }
            Self::Momentum {
                learning_rate,
                momentum,
                velocity,
            } => {
                let (learning_rate, momentum) = (*learning_rate, *momentum);
                for ((mut parameter, gradient), mut velocity) in
                    parameters.zip(velocity.tensors_mut())
                {
                    Zip::from(&mut parameter)
                        .and(&gradient)
                        .and(&mut velocity)
                        .for_each(|w, &g, v| {
                            // v = momentum * v - lr * g: the previous change,
                            // decayed, plus the plain SGD change.
                            *v = momentum * *v - learning_rate * g;
                            // w = w + v
                            *w += *v;
                        });
                }
            }
            Self::Adam {
                learning_rate,
                beta1,
                beta2,
                epsilon,
                first_moment,
                second_moment,
                step,
            } => {
                let (learning_rate, beta1, beta2, epsilon) =
                    (*learning_rate, *beta1, *beta2, *epsilon);
                *step += 1;
                // Moments start at 0, so after t steps they are biased towards
                // 0 by the factors 1 - beta^t; dividing by them removes the bias.
                let first_correction = 1.0 - beta1.powi(*step);
                let second_correction = 1.0 - beta2.powi(*step);
                for (((mut parameter, gradient), mut m), mut v) in parameters
                    .zip(first_moment.tensors_mut())
                    .zip(second_moment.tensors_mut())
                {
                    Zip::from(&mut parameter)
                        .and(&gradient)
                        .and(&mut m)
                        .and(&mut v)
                        .for_each(|w, &g, m, v| {
                            // m = beta1 * m + (1 - beta1) * g
                            *m = beta1 * *m + (1.0 - beta1) * g;
                            // v = beta2 * v + (1 - beta2) * g^2
                            *v = beta2 * *v + (1.0 - beta2) * g * g;
                            // w = w - lr * m_hat / (sqrt(v_hat) + epsilon)
                            let m_hat = *m / first_correction;
                            let v_hat = *v / second_correction;
                            *w -= learning_rate * m_hat / (v_hat.sqrt() + epsilon);
                        });
                }
            }
        }
    }
}

/// Panics unless `gradients` has one entry per layer of `network`, shaped
/// like its weights and bias.
fn assert_shapes_match(network: &Mlp, gradients: &Gradients) {
    let layers = network.layers();
    assert_eq!(
        gradients.layers.len(),
        layers.len(),
        "gradients must have one entry per layer"
    );
    for (index, (layer, gradient)) in layers.iter().zip(&gradients.layers).enumerate() {
        assert_eq!(
            gradient.weights.dim(),
            layer.weights().dim(),
            "weight gradient of layer {index} must be shaped like the weights"
        );
        assert_eq!(
            gradient.bias.as_ref().map(Array1::len),
            layer.bias().map(Array1::len),
            "bias gradient of layer {index} must be shaped like the bias"
        );
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, array};
    use rand::RngExt;

    use super::*;
    use crate::config::Loss;
    use crate::network::{Activation, Layer, LayerGradient};
    use crate::rng::{RngStream, rng};

    const ADAM: Optimizer = Optimizer::Adam {
        learning_rate: 0.1,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
    };

    /// 2 -> 1 (identity) with weights `[1, 2]` and bias `0.5`.
    fn network() -> Mlp {
        Mlp::from_layers(vec![Layer::from_parts(
            array![[1.0], [2.0]],
            Some(array![0.5]),
            Activation::Identity,
        )])
    }

    fn single_layer(weights: Array2<f64>, bias: f64) -> Gradients {
        Gradients {
            layers: vec![LayerGradient {
                weights,
                bias: Some(array![bias]),
            }],
        }
    }

    /// Weights and bias of the single layer of `network`.
    fn parameters(network: &Mlp) -> (Array2<f64>, f64) {
        let layer = &network.layers()[0];
        (layer.weights().clone(), layer.bias().unwrap()[0])
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-8, "{actual} != {expected}");
    }

    #[test]
    fn sgd_subtracts_scaled_gradient() {
        let mut network = network();
        let optimizer = Optimizer::Sgd { learning_rate: 0.5 };
        let mut state = OptimizerState::new(optimizer, &network);
        let gradients = single_layer(array![[0.5], [-1.0]], 0.25);

        state.step(&mut network, &gradients);
        assert_eq!(parameters(&network), (array![[0.75], [2.5]], 0.375));
        state.step(&mut network, &gradients);
        assert_eq!(parameters(&network), (array![[0.5], [3.0]], 0.25));
    }

    #[test]
    fn momentum_accumulates_velocity() {
        let mut network = network();
        let optimizer = Optimizer::Momentum {
            learning_rate: 0.5,
            momentum: 0.5,
        };
        let mut state = OptimizerState::new(optimizer, &network);
        let gradients = single_layer(array![[0.5], [-1.0]], 0.25);

        // v = -0.5 * g = [-0.25, 0.5], -0.125.
        state.step(&mut network, &gradients);
        assert_eq!(parameters(&network), (array![[0.75], [2.5]], 0.375));
        // v = 0.5 * v - 0.5 * g = [-0.375, 0.75], -0.1875.
        state.step(&mut network, &gradients);
        assert_eq!(parameters(&network), (array![[0.375], [3.25]], 0.1875));
        let OptimizerState::Momentum { velocity, .. } = &state else {
            panic!("expected the momentum state");
        };
        assert_eq!(velocity, &single_layer(array![[-0.375], [0.75]], -0.1875));
    }

    #[test]
    fn momentum_without_momentum_equals_sgd() {
        let mut generator = rng(7, RngStream::Init);
        let steps: Vec<Gradients> = (0..3)
            .map(|_| {
                let weights = Array2::from_shape_fn((2, 1), |_| generator.random_range(-1.0..1.0));
                single_layer(weights, generator.random_range(-1.0..1.0))
            })
            .collect();
        let run = |optimizer: Optimizer| {
            let mut network = network();
            let mut state = OptimizerState::new(optimizer, &network);
            for gradients in &steps {
                state.step(&mut network, gradients);
            }
            network
        };
        let sgd = run(Optimizer::Sgd { learning_rate: 0.3 });
        let momentum = run(Optimizer::Momentum {
            learning_rate: 0.3,
            momentum: 0.0,
        });
        assert_eq!(sgd, momentum);
    }

    #[test]
    fn adam_steps_by_learning_rate_for_constant_gradient() {
        let mut network = network();
        let mut state = OptimizerState::new(ADAM, &network);
        // With a constant gradient m_hat = g and v_hat = g^2, so every step
        // moves a parameter by lr * sign(g); a zero gradient moves nothing.
        let gradients = single_layer(array![[0.5], [0.0]], -2.0);
        for _ in 0..3 {
            state.step(&mut network, &gradients);
        }
        let (weights, bias) = parameters(&network);
        assert_close(weights[[0, 0]], 1.0 - 3.0 * 0.1);
        assert_eq!(weights[[1, 0]], 2.0);
        assert_close(bias, 0.5 + 3.0 * 0.1);
        let OptimizerState::Adam { step, .. } = state else {
            panic!("expected the Adam state");
        };
        assert_eq!(step, 3);
    }

    #[test]
    fn adam_corrects_moment_bias() {
        let mut network = network();
        let mut state = OptimizerState::new(ADAM, &network);
        state.step(&mut network, &single_layer(array![[1.0], [1.0]], 1.0));
        state.step(&mut network, &single_layer(array![[-1.0], [-1.0]], -1.0));
        // Step 1: m = 0.1, v = 0.001, m_hat = 1, v_hat = 1 -> w -= 0.1.
        // Step 2: m = 0.09 - 0.1 = -0.01, m_hat = -0.01 / 0.19 = -1/19;
        //         v = 0.000999 + 0.001 = 0.001999, v_hat = 0.001999 / 0.001999 = 1
        //         -> w += 0.1 / 19.
        let change = -0.1 + 0.1 / 19.0;
        let (weights, bias) = parameters(&network);
        assert_close(weights[[0, 0]], 1.0 + change);
        assert_close(weights[[1, 0]], 2.0 + change);
        assert_close(bias, 0.5 + change);
    }

    #[test]
    fn updates_layers_without_bias() {
        let gradients = Gradients {
            layers: vec![LayerGradient {
                weights: array![[1.0]],
                bias: None,
            }],
        };
        let momentum = Optimizer::Momentum {
            learning_rate: 0.5,
            momentum: 0.9,
        };
        for optimizer in [Optimizer::Sgd { learning_rate: 0.5 }, momentum, ADAM] {
            let mut network = Mlp::from_layers(vec![Layer::from_parts(
                array![[1.0]],
                None,
                Activation::Identity,
            )]);
            OptimizerState::new(optimizer, &network).step(&mut network, &gradients);
            assert!(network.layers()[0].weights()[[0, 0]] < 1.0, "{optimizer}");
            assert_eq!(network.layers()[0].bias(), None);
        }
    }

    #[test]
    #[should_panic(expected = "bias gradient of layer 0 must be shaped like the bias")]
    fn rejects_gradients_of_another_shape() {
        let mut network = network();
        let mut gradients = single_layer(array![[0.5], [-1.0]], 0.25);
        gradients.layers[0].bias = None;
        OptimizerState::new(ADAM, &network).step(&mut network, &gradients);
    }

    /// Full-batch training of `y = 2x - 1` with a single linear neuron.
    #[test]
    fn every_optimizer_fits_a_line() {
        let x = Array2::from_shape_fn((21, 1), |(row, _)| row as f64 / 10.0 - 1.0);
        let y = x.mapv(|x| 2.0 * x - 1.0);
        let optimizers = [
            Optimizer::Sgd { learning_rate: 0.1 },
            Optimizer::Momentum {
                learning_rate: 0.05,
                momentum: 0.9,
            },
            Optimizer::Adam {
                learning_rate: 0.05,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
        ];
        for optimizer in optimizers {
            let mut network = Mlp::from_layers(vec![Layer::from_parts(
                array![[0.0]],
                Some(array![0.0]),
                Activation::Identity,
            )]);
            let mut state = OptimizerState::new(optimizer, &network);
            let initial = network.loss(x.view(), &y, Loss::Mse);
            for _ in 0..300 {
                let cache = network.forward(x.view());
                let gradients = network.backward(&cache, &y, Loss::Mse);
                state.step(&mut network, &gradients);
            }
            let last = network.loss(x.view(), &y, Loss::Mse);
            assert!(last < 1e-4 * initial, "{optimizer}: {initial} -> {last}");
        }
    }
}
