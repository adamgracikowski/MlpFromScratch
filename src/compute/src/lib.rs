//! Compute layer of MlpFromScratch.
//!
//! Trains a multilayer perceptron implemented from scratch (forward pass,
//! backpropagation and SGD) and writes raw training artifacts to disk.
//! Plotting is left to the Python presentation layer.

pub mod artifacts;
pub mod cli;
pub mod config;
pub mod data;
pub mod network;
pub mod rng;
pub mod study;
pub mod training;
