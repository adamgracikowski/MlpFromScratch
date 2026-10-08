//! Weight update rule and its hyperparameters.

use serde::{Deserialize, Serialize};
use strum::Display;

/// Weight update rule and its hyperparameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Display)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Optimizer {
    /// Plain mini-batch stochastic gradient descent.
    #[strum(to_string = "sgd(lr={learning_rate})")]
    Sgd { learning_rate: f64 },
    /// SGD with classical momentum.
    #[strum(to_string = "momentum(lr={learning_rate}, momentum={momentum})")]
    Momentum { learning_rate: f64, momentum: f64 },
    /// Adam with bias-corrected moment estimates.
    #[strum(
        to_string = "adam(lr={learning_rate}, beta1={beta1}, beta2={beta2}, epsilon={epsilon:e})"
    )]
    Adam {
        learning_rate: f64,
        #[serde(default = "default_beta1")]
        beta1: f64,
        #[serde(default = "default_beta2")]
        beta2: f64,
        #[serde(default = "default_epsilon")]
        epsilon: f64,
    },
}

fn default_beta1() -> f64 {
    0.9
}

fn default_beta2() -> f64 {
    0.999
}

fn default_epsilon() -> f64 {
    1e-8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Result<Optimizer, toml::de::Error> {
        toml::from_str(text)
    }

    #[test]
    fn adam_parameters_have_defaults() {
        let optimizer = parse("kind = \"adam\"\nlearning_rate = 0.001\n").unwrap();
        assert_eq!(
            optimizer,
            Optimizer::Adam {
                learning_rate: 0.001,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            }
        );
    }

    #[test]
    fn parses_momentum() {
        let optimizer = parse("kind = \"momentum\"\nlearning_rate = 0.1\nmomentum = 0.9\n");
        assert_eq!(
            optimizer.unwrap(),
            Optimizer::Momentum {
                learning_rate: 0.1,
                momentum: 0.9,
            }
        );
    }

    #[test]
    fn rejects_parameter_of_another_variant() {
        assert!(parse("kind = \"sgd\"\nlearning_rate = 0.1\nmomentum = 0.9\n").is_err());
    }

    #[test]
    fn rejects_missing_learning_rate() {
        assert!(parse("kind = \"sgd\"\n").is_err());
    }

    #[test]
    fn displays_optimizer_with_parameters() {
        let sgd = parse("kind = \"sgd\"\nlearning_rate = 0.1\n").unwrap();
        assert_eq!(sgd.to_string(), "sgd(lr=0.1)");
        let momentum =
            parse("kind = \"momentum\"\nlearning_rate = 0.01\nmomentum = 0.9\n").unwrap();
        assert_eq!(momentum.to_string(), "momentum(lr=0.01, momentum=0.9)");
        let adam = parse("kind = \"adam\"\nlearning_rate = 0.001\n").unwrap();
        assert_eq!(
            adam.to_string(),
            "adam(lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)"
        );
    }
}
