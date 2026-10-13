//! Checks of the `[training]` section.

use crate::config::{DataConfig, EarlyStopping, Issue, TrainingConfig};

/// Checks the training length, batch size and early stopping; `[data]`
/// tells whether a training subset and a validation set are used.
pub(super) fn validate_training(
    training: &TrainingConfig,
    data: &DataConfig,
    issues: &mut Vec<Issue>,
) {
    if training.epochs == 0 {
        issues.push(Issue::error(&["training", "epochs"], "must be at least 1"));
    }
    if training.batch_size == 0 {
        issues.push(Issue::error(
            &["training", "batch_size"],
            "must be at least 1",
        ));
    }
    if let Some(subset) = data.train_subset.filter(|&subset| subset > 0) {
        if training.batch_size > subset {
            issues.push(Issue::warning(
                &["training", "batch_size"],
                format!(
                    "is larger than data.train_subset ({subset}), so every batch is the whole subset"
                ),
            ));
        }
    }
    if let Some(early_stopping) = &training.early_stopping {
        validate_early_stopping(early_stopping, data.validation_fraction, issues);
    }
}

/// Checks the early stopping settings; they need a validation set to monitor.
fn validate_early_stopping(
    early_stopping: &EarlyStopping,
    validation_fraction: f64,
    issues: &mut Vec<Issue>,
) {
    if early_stopping.patience == 0 {
        issues.push(Issue::error(
            &["training", "early_stopping", "patience"],
            "must be at least 1",
        ));
    }
    if !(early_stopping.min_delta >= 0.0 && early_stopping.min_delta.is_finite()) {
        issues.push(Issue::error(
            &["training", "early_stopping", "min_delta"],
            "must be non-negative",
        ));
    }
    if validation_fraction == 0.0 {
        issues.push(
            Issue::error(
                &["training", "early_stopping"],
                "needs a validation set to monitor",
            )
            .with_help("set data.validation_fraction > 0 or remove early_stopping"),
        );
    }
}

#[cfg(test)]
mod tests {
    use miette::Severity;

    use super::super::fixtures::{classification, errors, paths};
    use crate::config::{EarlyStopping, ExperimentConfig};

    #[test]
    fn rejects_zero_epochs_and_batch_size() {
        let mut config = classification();
        config.training.epochs = 0;
        config.training.batch_size = 0;
        assert_eq!(errors(&config), ["training.epochs", "training.batch_size"]);
    }

    #[test]
    fn warns_when_batch_exceeds_train_subset() {
        let mut config = classification();
        config.data.train_subset = Some(16);
        assert_eq!(paths(&config, Severity::Warning), ["training.batch_size"]);
    }

    fn with_early_stopping(patience: usize, min_delta: f64) -> ExperimentConfig {
        let mut config = classification();
        config.data.validation_fraction = 0.2;
        config.training.early_stopping = Some(EarlyStopping {
            patience,
            min_delta,
        });
        config
    }

    #[test]
    fn accepts_early_stopping_with_validation_set() {
        assert!(errors(&with_early_stopping(10, 1e-4)).is_empty());
    }

    #[test]
    fn rejects_early_stopping_out_of_range() {
        for min_delta in [-1e-4, f64::NAN, f64::INFINITY] {
            assert_eq!(
                errors(&with_early_stopping(0, min_delta)),
                [
                    "training.early_stopping.patience",
                    "training.early_stopping.min_delta"
                ]
            );
        }
    }

    #[test]
    fn rejects_early_stopping_without_validation_set() {
        let mut config = with_early_stopping(10, 0.0);
        config.data.validation_fraction = 0.0;
        assert_eq!(errors(&config), ["training.early_stopping"]);
    }
}
