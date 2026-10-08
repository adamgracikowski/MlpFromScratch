//! Compatibility of the task with the output activation, loss and target
//! normalization, which live in different sections.

use crate::config::{ExperimentConfig, Issue, Loss, Normalization, OutputActivation, Task};

/// Checks that the output activation, loss and target normalization match the task.
pub(super) fn validate_task(config: &ExperimentConfig, issues: &mut Vec<Issue>) {
    match config.task {
        Task::Regression => validate_regression(config, issues),
        Task::Classification => validate_classification(config, issues),
    }
}

/// Regression needs an unbounded (identity) output and a regression loss.
fn validate_regression(config: &ExperimentConfig, issues: &mut Vec<Issue>) {
    let output = config.model.output_activation;
    if output != OutputActivation::Identity {
        issues.push(
            Issue::error(
                &["model", "output_activation"],
                format!("{output} output cannot fit unbounded regression targets"),
            )
            .with_help("use \"identity\" for regression"),
        );
    }
    if config.training.loss == Loss::CrossEntropy {
        issues.push(
            Issue::error(
                &["training", "loss"],
                "cross_entropy is a classification loss",
            )
            .with_help("use \"mse\" or \"mae\" for regression"),
        );
    }
}

/// Classification needs raw labels, a classification loss and class scores
/// in `[0, 1]`; cross-entropy additionally needs a softmax output.
fn validate_classification(config: &ExperimentConfig, issues: &mut Vec<Issue>) {
    if config.data.target_normalization != Normalization::None {
        issues.push(
            Issue::error(
                &["data", "target_normalization"],
                "class labels cannot be normalized",
            )
            .with_help("use \"none\" for classification"),
        );
    }

    let output = config.model.output_activation;
    match config.training.loss {
        Loss::Mae => issues.push(
            Issue::error(&["training", "loss"], "mae is a regression loss")
                .with_help("use \"cross_entropy\" or \"mse\" for classification"),
        ),
        Loss::CrossEntropy if output != OutputActivation::Softmax => issues.push(
            Issue::error(
                &["model", "output_activation"],
                format!("cross_entropy requires softmax output, not {output}"),
            )
            .with_help("use \"softmax\" with cross_entropy"),
        ),
        Loss::Mse if output == OutputActivation::Identity => issues.push(
            Issue::error(
                &["model", "output_activation"],
                "identity output does not produce class scores in [0, 1]",
            )
            .with_help("use \"softmax\" or \"sigmoid\" for classification"),
        ),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixtures::{classification, errors, regression};
    use super::*;

    #[test]
    fn regression_requires_identity_output() {
        for output in [OutputActivation::Sigmoid, OutputActivation::Softmax] {
            let mut config = regression();
            config.model.output_activation = output;
            assert_eq!(errors(&config), ["model.output_activation"]);
        }
    }

    #[test]
    fn regression_rejects_cross_entropy() {
        let mut config = regression();
        config.training.loss = Loss::CrossEntropy;
        assert_eq!(errors(&config), ["training.loss"]);
    }

    #[test]
    fn regression_accepts_mae() {
        let mut config = regression();
        config.training.loss = Loss::Mae;
        assert!(errors(&config).is_empty());
    }

    #[test]
    fn classification_rejects_mae() {
        let mut config = classification();
        config.training.loss = Loss::Mae;
        assert_eq!(errors(&config), ["training.loss"]);
    }

    #[test]
    fn cross_entropy_requires_softmax() {
        for output in [OutputActivation::Sigmoid, OutputActivation::Identity] {
            let mut config = classification();
            config.model.output_activation = output;
            assert_eq!(errors(&config), ["model.output_activation"]);
        }
    }

    #[test]
    fn classification_mse_accepts_softmax_and_sigmoid() {
        for output in [OutputActivation::Softmax, OutputActivation::Sigmoid] {
            let mut config = classification();
            config.training.loss = Loss::Mse;
            config.model.output_activation = output;
            assert!(errors(&config).is_empty());
        }
    }

    #[test]
    fn classification_mse_rejects_identity() {
        let mut config = classification();
        config.training.loss = Loss::Mse;
        config.model.output_activation = OutputActivation::Identity;
        assert_eq!(errors(&config), ["model.output_activation"]);
    }

    #[test]
    fn classification_rejects_target_normalization() {
        let mut config = classification();
        config.data.target_normalization = Normalization::MinMax;
        assert_eq!(errors(&config), ["data.target_normalization"]);
    }
}
