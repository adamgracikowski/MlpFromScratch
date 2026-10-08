//! Experiment configuration: schema of `config.toml`, its validation and
//! loading with diagnostics that point at the offending line.
//!
//! Every field is required, with two exceptions: the Adam moment parameters
//! (`beta1`, `beta2`, `epsilon`) default to their standard values, and
//! `data.train_subset` may be omitted to train on the whole training set.
//! Enums with parameters use a `kind` tag, e.g.
//! `hidden_activation = { kind = "leaky_relu", alpha = 0.01 }`.

mod data;
mod diagnostic;
mod error;
#[cfg(test)]
mod examples;
mod experiment;
mod files;
mod issue;
mod load;
mod logging;
mod model;
mod optimizer;
mod output;
mod training;
mod validate;

pub use data::{DataConfig, Normalization};
pub use diagnostic::IssueDiagnostic;
pub use error::ConfigError;
pub use experiment::{ExperimentConfig, Task};
pub use files::{FileCheck, check_files};
pub use issue::{Issue, Key};
pub use load::{LoadedConfig, load_experiment, load_from_str};
pub use logging::LoggingConfig;
pub use model::{HiddenActivation, Init, ModelConfig, OutputActivation};
pub use optimizer::Optimizer;
pub use output::OutputConfig;
pub use training::{Loss, TrainingConfig};
pub use validate::{MAX_HIDDEN_LAYERS, validate};

/// A valid classification config used as the base of tests in other modules.
#[cfg(test)]
pub(crate) use experiment::tests::CLASSIFICATION;

/// Name of the configuration file inside an experiment directory.
pub const CONFIG_FILE_NAME: &str = "config.toml";
