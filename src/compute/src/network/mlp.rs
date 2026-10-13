//! The multilayer perceptron: a stack of fully connected layers.

use ndarray::{Array2, ArrayView2};
use rand::Rng;

use super::{Activation, ForwardCache, Layer, LayerCache};
use crate::config::{Loss, ModelConfig};

/// Multilayer perceptron: hidden layers followed by one output layer.
///
/// Without hidden layers it is a single layer mapping inputs to outputs.
#[derive(Debug, Clone, PartialEq)]
pub struct Mlp {
    layers: Vec<Layer>,
}

impl Mlp {
    /// Builds the network described by `model` for `input_size` features and
    /// `output_size` outputs (classes, or 1 for regression).
    ///
    /// Weights are drawn from `rng` layer by layer, from input to output, so
    /// the same generator state always gives the same network.
    pub fn new<R: Rng + ?Sized>(
        model: &ModelConfig,
        input_size: usize,
        output_size: usize,
        rng: &mut R,
    ) -> Self {
        let mut sizes = vec![input_size];
        sizes.extend(&model.hidden_layers);
        sizes.push(output_size);

        let output_index = sizes.len() - 2;
        let layers = sizes
            .windows(2)
            .enumerate()
            .map(|(index, pair)| {
                let activation = if index == output_index {
                    Activation::from(model.output_activation)
                } else {
                    Activation::from(model.hidden_activation)
                };
                Layer::new(pair[0], pair[1], model.bias, activation, model.init, rng)
            })
            .collect();
        Self::from_layers(layers)
    }

    /// Creates a network from given layers, e.g. hand-made ones in tests.
    ///
    /// Panics when there are no layers or when the output size of a layer
    /// differs from the input size of the next one.
    pub fn from_layers(layers: Vec<Layer>) -> Self {
        assert!(!layers.is_empty(), "a network needs at least one layer");
        for (index, pair) in layers.windows(2).enumerate() {
            assert_eq!(
                pair[0].output_size(),
                pair[1].input_size(),
                "layer {index} has {} output(s), but layer {} expects {} input(s)",
                pair[0].output_size(),
                index + 1,
                pair[1].input_size()
            );
        }
        Self { layers }
    }

    /// Layers from the first hidden layer to the output layer.
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    /// Number of input features.
    pub fn input_size(&self) -> usize {
        self.layers[0].input_size()
    }

    /// Number of outputs.
    pub fn output_size(&self) -> usize {
        self.layers[self.layers.len() - 1].output_size()
    }

    /// Number of trainable parameters of all layers.
    pub fn parameter_count(&self) -> usize {
        self.layers.iter().map(Layer::parameter_count).sum()
    }

    /// Runs the batch `input` (`n x d`) through the network and keeps `Z` and
    /// `A` of every layer for backpropagation.
    pub fn forward<'a>(&self, input: ArrayView2<'a, f64>) -> ForwardCache<'a> {
        let mut cache = ForwardCache {
            input,
            layers: Vec::with_capacity(self.layers.len()),
        };
        for (index, layer) in self.layers.iter().enumerate() {
            let (z, a) = layer.forward(cache.layer_input(index));
            cache.layers.push(LayerCache { z, a });
        }
        cache
    }

    /// Computes the outputs (`n x K`) for `input` (`n x d`) without keeping
    /// intermediate results, e.g. for evaluation or a prediction grid.
    pub fn predict(&self, input: ArrayView2<'_, f64>) -> Array2<f64> {
        let (first, rest) = self
            .layers
            .split_first()
            .expect("a network has at least one layer");
        let (_, mut activations) = first.forward(input);
        for layer in rest {
            (_, activations) = layer.forward(activations.view());
        }
        activations
    }

    /// Mean `loss` of the network on `input` (`n x d`) for `targets` (`n x K`).
    pub fn loss(&self, input: ArrayView2<'_, f64>, targets: &Array2<f64>, loss: Loss) -> f64 {
        let cache = self.forward(input);
        let output = cache
            .layers
            .last()
            .expect("a network has at least one layer");
        loss.value(&output.z, &output.a, targets)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, array};

    use super::*;
    use crate::config::{CLASSIFICATION, ExperimentConfig, HiddenActivation, OutputActivation};
    use crate::rng::{RngStream, rng};

    fn model() -> ModelConfig {
        let config: ExperimentConfig = toml::from_str(CLASSIFICATION).unwrap();
        config.model
    }

    fn network(model: &ModelConfig, seed: u64) -> Mlp {
        Mlp::new(model, 2, 3, &mut rng(seed, RngStream::Init))
    }

    /// 2 -> 2 (relu) -> 1 (identity) with hand-picked weights.
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
    fn builds_layers_from_config() {
        let model = model();
        let network = network(&model, 42);
        let shapes: Vec<(usize, usize)> = network
            .layers()
            .iter()
            .map(|layer| layer.weights().dim())
            .collect();
        assert_eq!(shapes, [(2, 16), (16, 16), (16, 3)]);
        let leaky_relu = Activation::LeakyRelu { alpha: 0.01 };
        let activations: Vec<Activation> = network.layers().iter().map(Layer::activation).collect();
        assert_eq!(activations, [leaky_relu, leaky_relu, Activation::Softmax]);
        assert_eq!((network.input_size(), network.output_size()), (2, 3));
        assert_eq!(
            network.parameter_count(),
            2 * 16 + 16 + 16 * 16 + 16 + 16 * 3 + 3
        );
    }

    #[test]
    fn builds_single_layer_without_hidden_layers() {
        let mut model = model();
        model.hidden_layers.clear();
        model.bias = false;
        model.output_activation = OutputActivation::Sigmoid;
        model.hidden_activation = HiddenActivation::Relu;
        let network = network(&model, 42);
        assert_eq!(network.layers().len(), 1);
        let layer = &network.layers()[0];
        assert_eq!(layer.weights().dim(), (2, 3));
        assert_eq!(layer.bias(), None);
        assert_eq!(layer.activation(), Activation::Sigmoid);
        assert_eq!(network.parameter_count(), 6);
    }

    #[test]
    fn same_seed_gives_same_network() {
        let model = model();
        assert_eq!(network(&model, 42), network(&model, 42));
        assert_ne!(network(&model, 42), network(&model, 43));
    }

    #[test]
    fn forward_matches_hand_computed_values() {
        let input = array![[1.0, 1.0], [0.0, -1.0]];
        let cache = hand_made().forward(input.view());
        assert_eq!(cache.input, input);
        // Hidden: [1, 1] -> Z [3, 1] -> A [3, 1]; [0, -1] -> Z [-2, 0] -> A [0, 0].
        assert_eq!(cache.layers[0].z, array![[3.0, 1.0], [-2.0, 0.0]]);
        assert_eq!(cache.layers[0].a, array![[3.0, 1.0], [0.0, 0.0]]);
        // Output: 3 - 2 + 0.5 and 0 - 0 + 0.5.
        assert_eq!(cache.layers[1].z, array![[1.5], [0.5]]);
        assert_eq!(cache.output(), &array![[1.5], [0.5]]);
        assert_eq!(cache.layer_input(1), cache.layers[0].a);
    }

    #[test]
    fn predict_matches_forward_output() {
        let model = model();
        let network = network(&model, 42);
        let input = Array2::from_shape_fn((5, 2), |(row, column)| {
            row as f64 * 0.3 - column as f64 * 0.7
        });
        let predictions = network.predict(input.view());
        assert_eq!(&predictions, network.forward(input.view()).output());
        assert_eq!(predictions.dim(), (5, 3));
        // Softmax output: every row is a probability vector.
        for sum in predictions.sum_axis(ndarray::Axis(1)) {
            assert!((sum - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn loss_uses_network_outputs() {
        let input = array![[1.0, 1.0], [0.0, -1.0]];
        // Outputs [1.5, 0.5] against [1, 1.5]: errors 0.5 and -1.
        let loss = hand_made().loss(input.view(), &array![[1.0], [1.5]], Loss::Mse);
        assert_eq!(loss, (0.25 + 1.0) / 2.0);
    }

    #[test]
    #[should_panic(expected = "layer expects 2 input column(s), got 3")]
    fn forward_rejects_wrong_input_width() {
        hand_made().forward(Array2::zeros((1, 3)).view());
    }

    #[test]
    #[should_panic(expected = "layer 0 has 2 output(s), but layer 1 expects 3 input(s)")]
    fn from_layers_rejects_mismatched_sizes() {
        Mlp::from_layers(vec![
            Layer::from_parts(Array2::zeros((2, 2)), None, Activation::Relu),
            Layer::from_parts(
                Array2::zeros((3, 1)),
                Some(Array1::zeros(1)),
                Activation::Identity,
            ),
        ]);
    }

    #[test]
    #[should_panic(expected = "at least one layer")]
    fn from_layers_rejects_empty_network() {
        Mlp::from_layers(Vec::new());
    }
}
