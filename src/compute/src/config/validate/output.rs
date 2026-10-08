//! Checks of the `[output]` section.

use crate::config::{Issue, OutputConfig};

/// Checks the prediction grid resolution and margin.
pub(super) fn validate_output(output: &OutputConfig, issues: &mut Vec<Issue>) {
    if output.grid_resolution < 2 {
        issues.push(Issue::error(
            &["output", "grid_resolution"],
            "must be at least 2",
        ));
    }
    if !(output.grid_margin >= 0.0 && output.grid_margin.is_finite()) {
        issues.push(Issue::error(
            &["output", "grid_margin"],
            "must be non-negative",
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixtures::{classification, errors};

    #[test]
    fn rejects_invalid_grid() {
        let mut config = classification();
        config.output.grid_resolution = 1;
        config.output.grid_margin = -0.1;
        assert_eq!(
            errors(&config),
            ["output.grid_resolution", "output.grid_margin"]
        );
    }
}
