//! Compute layer of MlpFromScratch.
//!
//! Trains a multilayer perceptron implemented from scratch (forward pass,
//! backpropagation and SGD) and writes raw training artifacts to disk.
//! Plotting is left to the Python presentation layer.

pub mod cli;
pub mod config;
pub mod data;
pub mod rng;
