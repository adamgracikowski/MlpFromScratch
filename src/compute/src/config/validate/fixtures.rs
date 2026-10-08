//! Valid configs and helpers shared by the validation tests.

use miette::Severity;

use super::validate;
use crate::config::experiment::tests::CLASSIFICATION;
use crate::config::{ExperimentConfig, Loss, Normalization, OutputActivation, Task};

/// A valid classification config: softmax, cross-entropy, Adam.
pub fn classification() -> ExperimentConfig {
    toml::from_str(CLASSIFICATION).unwrap()
}

/// A valid regression config derived from [`classification`].
pub fn regression() -> ExperimentConfig {
    let mut config = classification();
    config.task = Task::Regression;
    config.data.target_normalization = Normalization::Standard;
    config.model.output_activation = OutputActivation::Identity;
    config.training.loss = Loss::Mse;
    config
}

/// Paths of the issues with the given severity, in report order.
pub fn paths(config: &ExperimentConfig, severity: Severity) -> Vec<String> {
    validate(config)
        .into_iter()
        .filter(|issue| issue.severity == severity)
        .map(|issue| issue.path_string())
        .collect()
}

/// Paths of the errors, in report order.
pub fn errors(config: &ExperimentConfig) -> Vec<String> {
    paths(config, Severity::Error)
}
