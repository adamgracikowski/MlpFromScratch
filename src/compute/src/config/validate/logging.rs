//! Checks of the `[logging]` section.

use crate::config::{Issue, LoggingConfig};

/// Checks that every logging interval and limit is positive.
pub(super) fn validate_logging(logging: &LoggingConfig, issues: &mut Vec<Issue>) {
    let fields = [
        ("loss_every", logging.loss_every),
        ("weights_every", logging.weights_every),
        ("max_weights_per_layer", logging.max_weights_per_layer),
    ];
    for (name, value) in fields {
        if value == 0 {
            issues.push(Issue::error(&["logging", name], "must be at least 1"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixtures::{classification, errors};

    #[test]
    fn rejects_zero_logging_intervals() {
        let mut config = classification();
        config.logging.loss_every = 0;
        config.logging.weights_every = 0;
        config.logging.max_weights_per_layer = 0;
        assert_eq!(
            errors(&config),
            [
                "logging.loss_every",
                "logging.weights_every",
                "logging.max_weights_per_layer"
            ]
        );
    }
}
