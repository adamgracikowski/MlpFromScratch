//! Checks of the `[training]` section.

use crate::config::{Issue, TrainingConfig};

/// Checks the training length and batch size; `train_subset` comes from
/// `[data]` and only triggers a warning when it is smaller than a batch.
pub(super) fn validate_training(
    training: &TrainingConfig,
    train_subset: Option<usize>,
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
    if let Some(subset) = train_subset.filter(|&subset| subset > 0) {
        if training.batch_size > subset {
            issues.push(Issue::warning(
                &["training", "batch_size"],
                format!(
                    "is larger than data.train_subset ({subset}), so every batch is the whole subset"
                ),
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use miette::Severity;

    use super::super::fixtures::{classification, errors, paths};

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
}
