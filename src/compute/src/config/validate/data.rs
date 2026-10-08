//! Checks of the `[data]` section.

use crate::config::{DataConfig, Issue};

/// Checks the validation fraction and the training subset size.
pub(super) fn validate_data(data: &DataConfig, issues: &mut Vec<Issue>) {
    if !(0.0..1.0).contains(&data.validation_fraction) {
        issues.push(
            Issue::error(&["data", "validation_fraction"], "must be in [0, 1)")
                .with_help("use 0.0 to train on the whole training set without validation"),
        );
    }
    if data.train_subset == Some(0) {
        issues.push(
            Issue::error(&["data", "train_subset"], "must be at least 1")
                .with_help("omit the field to use the whole training set"),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixtures::{classification, errors};

    #[test]
    fn rejects_validation_fraction_out_of_range() {
        for fraction in [-0.1, 1.0, f64::NAN] {
            let mut config = classification();
            config.data.validation_fraction = fraction;
            assert_eq!(errors(&config), ["data.validation_fraction"]);
        }
    }

    #[test]
    fn rejects_empty_train_subset() {
        let mut config = classification();
        config.data.train_subset = Some(0);
        assert_eq!(errors(&config), ["data.train_subset"]);
    }
}
