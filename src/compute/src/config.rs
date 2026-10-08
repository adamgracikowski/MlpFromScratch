//! Experiment configuration: schema of `config.toml`.
//!
//! Every field is required, with two exceptions: the Adam moment parameters
//! (`beta1`, `beta2`, `epsilon`) default to their standard values, and
//! `data.train_subset` may be omitted to train on the whole training set.
//! Enums with parameters use a `kind` tag, e.g.
//! `hidden_activation = { kind = "leaky_relu", alpha = 0.01 }`.

mod data;
mod experiment;
mod logging;
mod model;
mod optimizer;
mod output;
mod training;

pub use data::{DataConfig, Normalization};
pub use experiment::{ExperimentConfig, Task};
pub use logging::LoggingConfig;
pub use model::{HiddenActivation, Init, ModelConfig, OutputActivation};
pub use optimizer::Optimizer;
pub use output::OutputConfig;
pub use training::{Loss, TrainingConfig};

/// Name of the configuration file inside an experiment directory.
pub const CONFIG_FILE_NAME: &str = "config.toml";
