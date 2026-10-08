//! A fully connected layer stored as a weight matrix and a bias vector.

use ndarray::{Array1, Array2, ArrayView2};
use rand::Rng;

use super::{Activation, initialize_weights};
use crate::config::Init;

/// Fully connected layer: `A = f(A_prev . W + b)`.
///
/// `W` is `in x out` (row `i` holds the weights leaving input `i`), `b` has
/// length `out` and is broadcast over the rows (samples) of the batch.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    weights: Array2<f64>,
    bias: Option<Array1<f64>>,
    activation: Activation,
}

impl Layer {
    /// Creates an `input_size x output_size` layer with weights drawn for
    /// `init` and, when `bias` is set, a zero bias vector.
    pub fn new<R: Rng + ?Sized>(
        input_size: usize,
        output_size: usize,
        bias: bool,
        activation: Activation,
        init: Init,
        rng: &mut R,
    ) -> Self {
        let weights = initialize_weights(init, input_size, output_size, rng);
        let bias = bias.then(|| Array1::zeros(output_size));
        Self::from_parts(weights, bias, activation)
    }

    /// Creates a layer from given parameters, e.g. hand-picked weights in tests.
    ///
    /// Panics when the bias length differs from the number of weight columns.
    pub fn from_parts(
        weights: Array2<f64>,
        bias: Option<Array1<f64>>,
        activation: Activation,
    ) -> Self {
        if let Some(bias) = &bias {
            assert_eq!(
                bias.len(),
                weights.ncols(),
                "bias length must equal the number of outputs"
            );
        }
        Self {
            weights,
            bias,
            activation,
        }
    }

    /// Weight matrix (`in x out`).
    pub fn weights(&self) -> &Array2<f64> {
        &self.weights
    }

    /// Bias vector (length `out`), `None` when the layer has no bias.
    pub fn bias(&self) -> Option<&Array1<f64>> {
        self.bias.as_ref()
    }

    /// Activation function of the layer.
    pub fn activation(&self) -> Activation {
        self.activation
    }

    /// Number of inputs (rows of `W`).
    pub fn input_size(&self) -> usize {
        self.weights.nrows()
    }

    /// Number of outputs (columns of `W`).
    pub fn output_size(&self) -> usize {
        self.weights.ncols()
    }

    /// Number of trainable parameters: weights plus biases.
    pub fn parameter_count(&self) -> usize {
        self.weights.len() + self.bias.as_ref().map_or(0, Array1::len)
    }

    /// Computes the pre-activations `Z = A_prev . W + b` (`n x out`) for a
    /// batch `input` (`n x in`).
    pub fn pre_activation(&self, input: ArrayView2<'_, f64>) -> Array2<f64> {
        assert_eq!(
            input.ncols(),
            self.input_size(),
            "layer expects {} input column(s), got {}",
            self.input_size(),
            input.ncols()
        );
        // (n x in) . (in x out) = n x out
        let mut z = input.dot(&self.weights);
        if let Some(bias) = &self.bias {
            // The bias row is added to every sample.
            z += bias;
        }
        assert_eq!(z.dim(), (input.nrows(), self.output_size()));
        z
    }

    /// Computes `(Z, A)` for a batch `input` (`n x in`); both are `n x out`.
    pub fn forward(&self, input: ArrayView2<'_, f64>) -> (Array2<f64>, Array2<f64>) {
        let z = self.pre_activation(input);
        let a = self.activation.forward(&z);
        (z, a)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;
    use crate::rng::{RngStream, rng};

    fn random_layer(bias: bool, seed: u64) -> Layer {
        let mut rng = rng(seed, RngStream::Init);
        Layer::new(3, 2, bias, Activation::Tanh, Init::Xavier, &mut rng)
    }

    fn hand_made_layer(bias: Option<Array1<f64>>) -> Layer {
        let weights = array![[1.0, 0.0, 2.0], [-1.0, 1.0, 0.0]];
        Layer::from_parts(weights, bias, Activation::Relu)
    }

    #[test]
    fn new_layer_has_expected_shapes() {
        let layer = random_layer(true, 42);
        assert_eq!(layer.weights().dim(), (3, 2));
        assert_eq!(layer.bias(), Some(&Array1::zeros(2)));
        assert_eq!((layer.input_size(), layer.output_size()), (3, 2));
        assert_eq!(layer.parameter_count(), 3 * 2 + 2);
        assert_eq!(layer.activation(), Activation::Tanh);
    }

    #[test]
    fn layer_without_bias_has_no_bias_parameters() {
        let layer = random_layer(false, 42);
        assert_eq!(layer.bias(), None);
        assert_eq!(layer.parameter_count(), 3 * 2);
    }

    #[test]
    fn same_seed_gives_same_layer() {
        assert_eq!(random_layer(true, 42), random_layer(true, 42));
        assert_ne!(random_layer(true, 42), random_layer(true, 43));
    }

    #[test]
    fn forward_matches_hand_computed_values() {
        let layer = hand_made_layer(Some(array![0.5, 0.0, -1.0]));
        let input = array![[1.0, -1.0], [0.0, 2.0]];
        let (z, a) = layer.forward(input.view());
        // [1, -1] -> [1 + 1 + 0.5, 0 - 1 + 0, 2 + 0 - 1]
        // [0,  2] -> [0 - 2 + 0.5, 0 + 2 + 0, 0 + 0 - 1]
        assert_eq!(z, array![[2.5, -1.0, 1.0], [-1.5, 2.0, -1.0]]);
        assert_eq!(a, array![[2.5, 0.0, 1.0], [0.0, 2.0, 0.0]]);
    }

    #[test]
    fn forward_without_bias_does_not_shift() {
        let layer = hand_made_layer(None);
        let z = layer.pre_activation(array![[1.0, -1.0]].view());
        assert_eq!(z, array![[2.0, -1.0, 2.0]]);
    }

    #[test]
    #[should_panic(expected = "layer expects 2 input column(s), got 3")]
    fn forward_rejects_wrong_input_width() {
        hand_made_layer(None).forward(array![[1.0, 2.0, 3.0]].view());
    }

    #[test]
    #[should_panic(expected = "bias length must equal the number of outputs")]
    fn from_parts_rejects_mismatched_bias() {
        hand_made_layer(Some(array![1.0, 2.0]));
    }
}
