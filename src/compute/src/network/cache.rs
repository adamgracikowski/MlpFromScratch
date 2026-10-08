//! Intermediate results of a forward pass, kept for backpropagation.

use ndarray::{Array2, ArrayView2};

/// Pre-activations and activations of one layer for a batch (`n x out`).
#[derive(Debug, Clone, PartialEq)]
pub struct LayerCache {
    /// Pre-activations `Z = A_prev . W + b`.
    pub z: Array2<f64>,
    /// Activations `A = f(Z)`.
    pub a: Array2<f64>,
}

/// Everything backpropagation needs from a forward pass over a batch.
///
/// The batch itself is borrowed rather than copied. It is the input `A_0` of
/// the first layer.
#[derive(Debug, Clone, PartialEq)]
pub struct ForwardCache<'a> {
    /// Input batch (`n x d`).
    pub input: ArrayView2<'a, f64>,
    /// One entry per layer, from the first hidden layer to the output layer.
    pub layers: Vec<LayerCache>,
}

impl ForwardCache<'_> {
    /// Input of layer `index`: the batch for the first layer, otherwise the
    /// activations of the previous layer.
    pub fn layer_input(&self, index: usize) -> ArrayView2<'_, f64> {
        match index {
            0 => self.input.view(),
            _ => self.layers[index - 1].a.view(),
        }
    }

    /// Output of the network (`n x K`): activations of the last layer.
    pub fn output(&self) -> &Array2<f64> {
        &self
            .layers
            .last()
            .expect("a network has at least one layer")
            .a
    }
}
