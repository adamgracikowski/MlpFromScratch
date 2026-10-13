//! Gradients of the loss with respect to the network parameters.

use ndarray::{Array1, Array2, ArrayViewD, ArrayViewMutD};

use super::{Layer, Mlp};

/// Gradient of the loss with respect to the parameters of one layer.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerGradient {
    /// `dL/dW` (`in x out`, like the weights).
    pub weights: Array2<f64>,
    /// `dL/db` (length `out`), `None` when the layer has no bias.
    pub bias: Option<Array1<f64>>,
}

impl LayerGradient {
    /// Zeros shaped like the parameters of `layer`, e.g. an optimizer buffer.
    pub fn zeros_like(layer: &Layer) -> Self {
        Self {
            weights: Array2::zeros(layer.weights().dim()),
            bias: layer.bias().map(|bias| Array1::zeros(bias.len())),
        }
    }

    /// Views of the weight gradient and, if present, the bias gradient.
    pub fn tensors(&self) -> impl Iterator<Item = ArrayViewD<'_, f64>> {
        let bias = self.bias.as_ref().map(|bias| bias.view().into_dyn());
        std::iter::once(self.weights.view().into_dyn()).chain(bias)
    }

    /// Mutable views of the weight gradient and, if present, the bias gradient.
    pub fn tensors_mut(&mut self) -> impl Iterator<Item = ArrayViewMutD<'_, f64>> {
        let weights = self.weights.view_mut().into_dyn();
        let bias = self.bias.as_mut().map(|bias| bias.view_mut().into_dyn());
        std::iter::once(weights).chain(bias)
    }
}

/// Gradients of all layers, from the first hidden layer to the output layer.
#[derive(Debug, Clone, PartialEq)]
pub struct Gradients {
    pub layers: Vec<LayerGradient>,
}

impl Gradients {
    /// Zeros shaped like every parameter of `network`.
    pub fn zeros_like(network: &Mlp) -> Self {
        Self {
            layers: network
                .layers()
                .iter()
                .map(LayerGradient::zeros_like)
                .collect(),
        }
    }

    /// Views of all gradient tensors in the order of
    /// [`Mlp::parameters_mut`]: per layer the weights, then the bias.
    pub fn tensors(&self) -> impl Iterator<Item = ArrayViewD<'_, f64>> {
        self.layers.iter().flat_map(LayerGradient::tensors)
    }

    /// Mutable views of all gradient tensors, ordered like [`Gradients::tensors`].
    pub fn tensors_mut(&mut self) -> impl Iterator<Item = ArrayViewMutD<'_, f64>> {
        self.layers.iter_mut().flat_map(LayerGradient::tensors_mut)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::Activation;

    fn network() -> Mlp {
        Mlp::from_layers(vec![
            Layer::from_parts(Array2::ones((3, 4)), None, Activation::Tanh),
            Layer::from_parts(
                Array2::ones((4, 2)),
                Some(Array1::ones(2)),
                Activation::Identity,
            ),
        ])
    }

    #[test]
    fn zeros_like_matches_network_shapes() {
        let gradients = Gradients::zeros_like(&network());
        assert_eq!(gradients.layers[0].weights, Array2::zeros((3, 4)));
        assert_eq!(gradients.layers[0].bias, None);
        assert_eq!(gradients.layers[1].weights, Array2::zeros((4, 2)));
        assert_eq!(gradients.layers[1].bias, Some(Array1::zeros(2)));
    }

    #[test]
    fn tensors_follow_parameter_order() {
        let mut network = network();
        let mut gradients = Gradients::zeros_like(&network);
        let parameters: Vec<Vec<usize>> = network
            .parameters_mut()
            .map(|parameter| parameter.shape().to_vec())
            .collect();
        let tensors: Vec<Vec<usize>> = gradients
            .tensors()
            .map(|tensor| tensor.shape().to_vec())
            .collect();
        assert_eq!(tensors, parameters);
        assert_eq!(tensors, [vec![3, 4], vec![4, 2], vec![2]]);

        for mut tensor in gradients.tensors_mut() {
            tensor.fill(1.0);
        }
        assert_eq!(gradients.layers[1].bias, Some(Array1::ones(2)));
    }
}
