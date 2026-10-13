//! Multilayer perceptron: layers as matrices, activations, the forward pass
//! and losses.

mod activation;
mod cache;
mod init;
mod layer;
mod loss;
mod mlp;

pub use activation::Activation;
pub use cache::{ForwardCache, LayerCache};
pub use init::initialize_weights;
pub use layer::Layer;
pub use mlp::Mlp;
