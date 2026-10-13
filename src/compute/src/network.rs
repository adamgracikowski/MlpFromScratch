//! Multilayer perceptron: layers as matrices, activations, the forward pass,
//! losses and backpropagation.

mod activation;
mod backward;
mod cache;
mod gradients;
mod init;
mod layer;
mod loss;
mod mlp;

pub use activation::Activation;
pub use cache::{ForwardCache, LayerCache};
pub use gradients::{Gradients, LayerGradient};
pub use init::initialize_weights;
pub use layer::Layer;
pub use mlp::Mlp;
