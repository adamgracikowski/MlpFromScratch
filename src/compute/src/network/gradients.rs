//! Gradients of the loss with respect to the network parameters.

use ndarray::{Array1, Array2};

/// Gradient of the loss with respect to the parameters of one layer.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerGradient {
    /// `dL/dW` (`in x out`, like the weights).
    pub weights: Array2<f64>,
    /// `dL/db` (length `out`), `None` when the layer has no bias.
    pub bias: Option<Array1<f64>>,
}

/// Gradients of all layers, from the first hidden layer to the output layer.
#[derive(Debug, Clone, PartialEq)]
pub struct Gradients {
    pub layers: Vec<LayerGradient>,
}
