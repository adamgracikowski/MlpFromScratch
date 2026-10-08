//! Semantic validation of a parsed config, without any I/O.
//!
//! Each config section has its own checks in `validate/`; `task.rs` checks
//! the compatibility of the task, output activation and loss across sections.

mod data;
#[cfg(test)]
mod fixtures;
mod logging;
mod model;
mod optimizer;
mod output;
mod task;
mod training;

use super::{ExperimentConfig, Issue};

pub use model::MAX_HIDDEN_LAYERS;

/// Checks value ranges and the compatibility of task, output activation and loss.
///
/// Returns every problem found, errors and warnings alike, so that all of them
/// can be reported at once.
pub fn validate(config: &ExperimentConfig) -> Vec<Issue> {
    let mut issues = Vec::new();
    data::validate_data(&config.data, &mut issues);
    model::validate_model(&config.model, &mut issues);
    task::validate_task(config, &mut issues);
    optimizer::validate_optimizer(&config.optimizer, &mut issues);
    training::validate_training(&config.training, config.data.train_subset, &mut issues);
    logging::validate_logging(&config.logging, &mut issues);
    output::validate_output(&config.output, &mut issues);
    issues
}

#[cfg(test)]
mod tests {
    use super::fixtures::{classification, errors, regression};
    use super::*;
    use crate::config::OutputActivation;

    #[test]
    fn valid_configs_have_no_issues() {
        assert!(validate(&classification()).is_empty());
        assert!(validate(&regression()).is_empty());
    }

    #[test]
    fn reports_all_errors_at_once() {
        let mut config = regression();
        config.model.output_activation = OutputActivation::Softmax;
        config.training.epochs = 0;
        config.output.grid_resolution = 0;
        assert_eq!(
            errors(&config),
            [
                "model.output_activation",
                "training.epochs",
                "output.grid_resolution"
            ]
        );
    }
}
