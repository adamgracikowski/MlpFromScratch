//! Checks of the `[model]` section.

use miette::Severity;

use crate::config::{HiddenActivation, Init, Issue, ModelConfig};

/// Maximum number of hidden layers the experiments use.
pub const MAX_HIDDEN_LAYERS: usize = 4;

/// Checks the architecture, activation and initialization parameters, and
/// warns about an initialization that does not suit the hidden activation.
pub(super) fn validate_model(model: &ModelConfig, issues: &mut Vec<Issue>) {
    if model.hidden_layers.len() > MAX_HIDDEN_LAYERS {
        issues.push(Issue::error(
            &["model", "hidden_layers"],
            format!(
                "has {} layers, at most {MAX_HIDDEN_LAYERS} are supported",
                model.hidden_layers.len()
            ),
        ));
    }
    for (index, &size) in model.hidden_layers.iter().enumerate() {
        if size == 0 {
            issues.push(
                Issue::error(&["model", "hidden_layers"], "layer size must be at least 1")
                    .at_index(index),
            );
        }
    }

    if let HiddenActivation::LeakyRelu { alpha } = model.hidden_activation {
        if !(alpha > 0.0 && alpha < 1.0) {
            issues.push(Issue::error(
                &["model", "hidden_activation", "alpha"],
                "must be in (0, 1)",
            ));
        }
    }
    if let Init::Uniform { limit } = model.init {
        if !(limit > 0.0 && limit.is_finite()) {
            issues.push(Issue::error(
                &["model", "init", "limit"],
                "must be positive",
            ));
        }
    }

    if model.hidden_layers.is_empty() {
        issues.push(Issue::new(
            Severity::Advice,
            &["model", "hidden_activation"],
            "is unused because the network has no hidden layers",
        ));
    } else if let Some(suggested) = unusual_init(model.init, model.hidden_activation) {
        issues.push(
            Issue::warning(
                &["model", "init"],
                format!(
                    "{} initialization is unusual for {} activation",
                    model.init, model.hidden_activation
                ),
            )
            .with_help(format!("{suggested} initialization is the usual choice")),
        );
    }
}

/// Returns the usual initialization when `init` does not suit `activation`.
fn unusual_init(init: Init, activation: HiddenActivation) -> Option<&'static str> {
    let relu_like = matches!(
        activation,
        HiddenActivation::Relu | HiddenActivation::LeakyRelu { .. }
    );
    match init {
        Init::Xavier if relu_like => Some("he"),
        Init::He if !relu_like => Some("xavier"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use miette::Severity;

    use super::super::fixtures::{classification, errors, paths};
    use super::*;

    #[test]
    fn rejects_too_many_hidden_layers() {
        let mut config = classification();
        config.model.hidden_layers = vec![8; MAX_HIDDEN_LAYERS + 1];
        assert_eq!(errors(&config), ["model.hidden_layers"]);
    }

    #[test]
    fn accepts_max_hidden_layers() {
        let mut config = classification();
        config.model.hidden_layers = vec![8; MAX_HIDDEN_LAYERS];
        assert!(errors(&config).is_empty());
    }

    #[test]
    fn rejects_empty_hidden_layer() {
        let mut config = classification();
        config.model.hidden_layers = vec![8, 0];
        assert_eq!(errors(&config), ["model.hidden_layers[1]"]);
    }

    #[test]
    fn notes_unused_hidden_activation() {
        let mut config = classification();
        config.model.hidden_layers = vec![];
        assert!(errors(&config).is_empty());
        assert_eq!(
            paths(&config, Severity::Advice),
            ["model.hidden_activation"]
        );
        // Without hidden layers the init/activation pairing does not matter.
        config.model.init = Init::Xavier;
        assert!(paths(&config, Severity::Warning).is_empty());
    }

    #[test]
    fn rejects_leaky_relu_alpha_out_of_range() {
        for alpha in [0.0, 1.0, -0.5] {
            let mut config = classification();
            config.model.hidden_activation = HiddenActivation::LeakyRelu { alpha };
            assert_eq!(errors(&config), ["model.hidden_activation.alpha"]);
        }
    }

    #[test]
    fn rejects_non_positive_uniform_limit() {
        let mut config = classification();
        config.model.init = Init::Uniform { limit: 0.0 };
        assert_eq!(errors(&config), ["model.init.limit"]);
    }

    #[test]
    fn warns_about_unusual_init() {
        let mut config = classification();
        config.model.init = Init::Xavier;
        assert_eq!(paths(&config, Severity::Warning), ["model.init"]);

        config.model.hidden_activation = HiddenActivation::Tanh;
        assert!(paths(&config, Severity::Warning).is_empty());
        config.model.init = Init::He;
        assert_eq!(paths(&config, Severity::Warning), ["model.init"]);
    }
}
