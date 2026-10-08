//! Multilayer perceptron: layers as matrices, activations and the forward pass.

mod activation;
mod init;
mod layer;

pub use activation::Activation;
pub use init::initialize_weights;
pub use layer::Layer;
