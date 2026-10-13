//! Backpropagation: gradients of the loss with respect to every parameter.

use ndarray::{Array2, Axis};

use super::{ForwardCache, Gradients, LayerGradient, Mlp};
use crate::config::Loss;

impl Mlp {
    /// Computes the gradients of the mean `loss` of a batch with respect to
    /// all weights and biases, from the `cache` of `forward` on that batch
    /// and its `targets` (`n x K`).
    pub fn backward(
        &self,
        cache: &ForwardCache<'_>,
        targets: &Array2<f64>,
        loss: Loss,
    ) -> Gradients {
        let layers = self.layers();
        assert_eq!(
            cache.layers.len(),
            layers.len(),
            "cache must come from a forward pass of this network"
        );
        let samples = cache.input.nrows();
        let last = layers.len() - 1;

        // Output layer: dL/dZ comes straight from the loss (n x K).
        let output = &cache.layers[last];
        let mut grad_z =
            loss.output_gradient(layers[last].activation(), &output.z, &output.a, targets);

        let mut gradients = Vec::with_capacity(layers.len());
        for index in (0..layers.len()).rev() {
            let layer = &layers[index];
            let input = cache.layer_input(index);
            assert_eq!(grad_z.dim(), (samples, layer.output_size()));

            // dW = A_prev^T . dZ: (in x n) . (n x out) = in x out.
            // The sum over samples is already the batch mean, since dZ
            // carries the 1/n of the loss.
            let grad_weights = input.t().dot(&grad_z);
            assert_eq!(grad_weights.dim(), layer.weights().dim());

            // db = column sums of dZ: the bias is added to every sample (length out).
            let grad_bias = layer.bias().map(|_| grad_z.sum_axis(Axis(0)));

            if index > 0 {
                // dA_prev = dZ . W^T: (n x out) . (out x in) = n x in.
                let grad_input = grad_z.dot(&layer.weights().t());
                assert_eq!(grad_input.dim(), (samples, layer.input_size()));

                // dZ_prev = VJP of the previous layer's activation applied to dA_prev.
                let previous = &cache.layers[index - 1];
                grad_z =
                    layers[index - 1]
                        .activation()
                        .backward(&previous.z, &previous.a, &grad_input);
            }

            gradients.push(LayerGradient {
                weights: grad_weights,
                bias: grad_bias,
            });
        }

        // Collected from the output layer backwards.
        gradients.reverse();
        Gradients { layers: gradients }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, array};
    use rand::RngExt;

    use super::*;
    use crate::config::Init;
    use crate::network::{Activation, Layer};
    use crate::rng::{RngStream, rng};

    const HIDDEN: [Activation; 4] = [
        Activation::Sigmoid,
        Activation::Tanh,
        Activation::Relu,
        Activation::LeakyRelu { alpha: 0.1 },
    ];

    /// Every (loss, output activation) pair allowed by config validation.
    const OUTPUTS: [(Loss, Activation); 5] = [
        (Loss::CrossEntropy, Activation::Softmax),
        (Loss::Mse, Activation::Softmax),
        (Loss::Mse, Activation::Sigmoid),
        (Loss::Mse, Activation::Identity),
        (Loss::Mae, Activation::Identity),
    ];

    /// Hidden layer sizes and bias flag.
    const ARCHITECTURES: [(&[usize], bool); 4] =
        [(&[], true), (&[4], true), (&[4, 3], true), (&[4, 3], false)];

    const INPUTS: usize = 3;
    const OUTPUT_SIZE: usize = 3;
    const SAMPLES: usize = 5;
    const STEP: f64 = 1e-5;
    /// Minimal distance from a kink (ReLU at `z = 0`, MAE at `a = y`).
    const KINK_MARGIN: f64 = 1e-3;

    /// A parameter of one layer.
    #[derive(Clone, Copy)]
    enum Parameter {
        Weight(usize, usize),
        Bias(usize),
    }

    struct Case {
        network: Mlp,
        input: Array2<f64>,
        targets: Array2<f64>,
        loss: Loss,
    }

    /// Random network and batch for one combination.
    fn random_case(
        hidden: Activation,
        (loss, output): (Loss, Activation),
        (sizes, bias): (&[usize], bool),
        seed: u64,
    ) -> Case {
        let mut init_rng = rng(seed, RngStream::Init);
        let mut all_sizes = vec![INPUTS];
        all_sizes.extend(sizes);
        all_sizes.push(OUTPUT_SIZE);
        let last = all_sizes.len() - 2;
        let layers = all_sizes
            .windows(2)
            .enumerate()
            .map(|(index, pair)| {
                let activation = if index == last { output } else { hidden };
                let init = Init::Uniform { limit: 1.0 };
                let mut layer = Layer::new(pair[0], pair[1], bias, activation, init, &mut init_rng);
                // Non-zero biases, so their effect is checked too.
                if bias {
                    let bias = Array1::from_shape_fn(pair[1], |_| init_rng.random_range(-0.5..0.5));
                    layer = Layer::from_parts(layer.weights().clone(), Some(bias), activation);
                }
                layer
            })
            .collect();

        let mut data_rng = rng(seed, RngStream::Data);
        let input = Array2::from_shape_fn((SAMPLES, INPUTS), |_| data_rng.random_range(-1.0..1.0));
        let targets = match output {
            Activation::Identity => {
                Array2::from_shape_fn((SAMPLES, OUTPUT_SIZE), |_| data_rng.random_range(-2.0..2.0))
            }
            _ => {
                let mut targets = Array2::zeros((SAMPLES, OUTPUT_SIZE));
                for mut row in targets.rows_mut() {
                    row[data_rng.random_range(0..OUTPUT_SIZE)] = 1.0;
                }
                targets
            }
        };
        Case {
            network: Mlp::from_layers(layers),
            input,
            targets,
            loss,
        }
    }

    /// Whether some value sits so close to a kink that central differences
    /// would cross it.
    fn near_kink(case: &Case) -> bool {
        let cache = case.network.forward(case.input.view());
        let close = |value: f64| value.abs() < KINK_MARGIN;
        let hidden_kink =
            case.network
                .layers()
                .iter()
                .zip(&cache.layers)
                .any(|(layer, layer_cache)| {
                    matches!(
                        layer.activation(),
                        Activation::Relu | Activation::LeakyRelu { .. }
                    ) && layer_cache.z.iter().any(|&z| close(z))
                });
        let mae_kink = case.loss == Loss::Mae
            && (cache.output() - &case.targets)
                .iter()
                .any(|&difference| close(difference));
        hidden_kink || mae_kink
    }

    /// First case (by seed) with no value near a kink, so the check is deterministic.
    fn smooth_case(
        hidden: Activation,
        output: (Loss, Activation),
        architecture: (&[usize], bool),
    ) -> Case {
        (0..100)
            .map(|seed| random_case(hidden, output, architecture, seed))
            .find(|case| !near_kink(case))
            .expect("some seed avoids every kink")
    }

    /// Copy of `network` with one parameter of layer `layer` shifted by `delta`.
    fn shifted(network: &Mlp, layer: usize, parameter: Parameter, delta: f64) -> Mlp {
        let layers = network
            .layers()
            .iter()
            .enumerate()
            .map(|(index, original)| {
                let mut weights = original.weights().clone();
                let mut bias = original.bias().cloned();
                if index == layer {
                    match parameter {
                        Parameter::Weight(row, column) => weights[[row, column]] += delta,
                        Parameter::Bias(column) => {
                            bias.as_mut().expect("layer has a bias")[column] += delta
                        }
                    }
                }
                Layer::from_parts(weights, bias, original.activation())
            })
            .collect();
        Mlp::from_layers(layers)
    }

    /// Central-difference estimate of every parameter's gradient.
    fn numerical_gradients(case: &Case) -> Gradients {
        let derivative = |layer: usize, parameter: Parameter| {
            let loss = |delta: f64| {
                shifted(&case.network, layer, parameter, delta).loss(
                    case.input.view(),
                    &case.targets,
                    case.loss,
                )
            };
            let (plus, minus) = (loss(STEP), loss(-STEP));
            (plus - minus) / (2.0 * STEP)
        };
        let layers = case
            .network
            .layers()
            .iter()
            .enumerate()
            .map(|(index, layer)| LayerGradient {
                weights: Array2::from_shape_fn(layer.weights().dim(), |(row, column)| {
                    derivative(index, Parameter::Weight(row, column))
                }),
                bias: layer.bias().map(|bias| {
                    Array1::from_shape_fn(bias.len(), |column| {
                        derivative(index, Parameter::Bias(column))
                    })
                }),
            })
            .collect();
        Gradients { layers }
    }

    /// Relative error with an absolute floor for gradients that are ~0,
    /// where central differences only leave rounding noise.
    fn assert_close(analytical: f64, numerical: f64, context: &str) {
        let error = (analytical - numerical).abs();
        let scale = analytical.abs() + numerical.abs();
        assert!(
            error <= 1e-6 * scale + 1e-9,
            "{context}: analytical {analytical}, numerical {numerical}"
        );
    }

    fn assert_gradients_close(analytical: &Gradients, numerical: &Gradients, context: &str) {
        assert_eq!(analytical.layers.len(), numerical.layers.len());
        for (index, (actual, expected)) in
            analytical.layers.iter().zip(&numerical.layers).enumerate()
        {
            assert_eq!(actual.weights.dim(), expected.weights.dim());
            for (&a, &n) in actual.weights.iter().zip(&expected.weights) {
                assert_close(a, n, &format!("{context}, layer {index} weights"));
            }
            assert_eq!(actual.bias.is_some(), expected.bias.is_some());
            if let (Some(actual), Some(expected)) = (&actual.bias, &expected.bias) {
                for (&a, &n) in actual.iter().zip(expected) {
                    assert_close(a, n, &format!("{context}, layer {index} bias"));
                }
            }
        }
    }

    #[test]
    fn backward_matches_numerical_gradient() {
        for hidden in HIDDEN {
            for output in OUTPUTS {
                for architecture in ARCHITECTURES {
                    let case = smooth_case(hidden, output, architecture);
                    let cache = case.network.forward(case.input.view());
                    let analytical = case.network.backward(&cache, &case.targets, case.loss);
                    let numerical = numerical_gradients(&case);
                    let context = format!(
                        "hidden {hidden}, output {}, loss {}, layers {:?}, bias {}",
                        output.1, output.0, architecture.0, architecture.1
                    );
                    assert_gradients_close(&analytical, &numerical, &context);
                }
            }
        }
    }

    /// 2 -> 2 (relu) -> 1 (identity), the hand-made network of `mlp.rs`.
    fn hand_made() -> Mlp {
        Mlp::from_layers(vec![
            Layer::from_parts(
                array![[1.0, -1.0], [2.0, 1.0]],
                Some(array![0.0, 1.0]),
                Activation::Relu,
            ),
            Layer::from_parts(
                array![[1.0], [-2.0]],
                Some(array![0.5]),
                Activation::Identity,
            ),
        ])
    }

    #[test]
    fn backward_matches_hand_computed_gradients() {
        let input = array![[1.0, 1.0], [0.0, -1.0]];
        let targets = array![[1.0], [1.5]];
        let network = hand_made();
        let cache = network.forward(input.view());
        let gradients = network.backward(&cache, &targets, Loss::Mse);
        // Outputs [1.5, 0.5] -> dZ2 = 2 (A - Y) / 2 = [0.5, -1.0].
        // Hidden A1 = [[3, 1], [0, 0]] -> dW2 = A1^T . dZ2 = [1.5, 0.5], db2 = -0.5.
        assert_eq!(gradients.layers[1].weights, array![[1.5], [0.5]]);
        assert_eq!(gradients.layers[1].bias, Some(array![-0.5]));
        // dA1 = dZ2 . W2^T = [[0.5, -1], [-1, 2]]; Z1 = [[3, 1], [-2, 0]]
        // -> dZ1 = [[0.5, -1], [0, 0]] (ReLU blocks z <= 0).
        // dW1 = X^T . dZ1 = [[0.5, -1], [0.5, -1]], db1 = [0.5, -1].
        assert_eq!(
            gradients.layers[0].weights,
            array![[0.5, -1.0], [0.5, -1.0]]
        );
        assert_eq!(gradients.layers[0].bias, Some(array![0.5, -1.0]));
    }

    #[test]
    fn layers_without_bias_have_no_bias_gradient() {
        let case = smooth_case(Activation::Tanh, OUTPUTS[0], ARCHITECTURES[3]);
        let cache = case.network.forward(case.input.view());
        let gradients = case.network.backward(&cache, &case.targets, case.loss);
        assert!(gradients.layers.iter().all(|layer| layer.bias.is_none()));
    }

    #[test]
    #[should_panic(expected = "cache must come from a forward pass of this network")]
    fn backward_rejects_foreign_cache() {
        let network = hand_made();
        let input = array![[1.0, 1.0]];
        let mut cache = network.forward(input.view());
        cache.layers.pop();
        network.backward(&cache, &array![[1.0]], Loss::Mse);
    }
}
