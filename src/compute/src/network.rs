//! Multilayer perceptron: layers as matrices, activations and the forward pass.

mod activation;
mod cache;
mod init;
mod layer;
mod mlp;

pub use activation::Activation;
pub use cache::{ForwardCache, LayerCache};
pub use init::initialize_weights;
pub use layer::Layer;
pub use mlp::Mlp;
