//! Weight update rule and its hyperparameters.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Weight update rule and its hyperparameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Optimizer {
    /// Plain mini-batch stochastic gradient descent.
    Sgd { learning_rate: f64 },
    /// SGD with classical momentum.
    Momentum { learning_rate: f64, momentum: f64 },
    /// Adam with bias-corrected moment estimates.
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

impl fmt::Display for Optimizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Optimizer::Sgd { learning_rate } => write!(f, "sgd(lr={learning_rate})"),
            Optimizer::Momentum {
                learning_rate,
                momentum,
            } => write!(f, "momentum(lr={learning_rate}, momentum={momentum})"),
            Optimizer::Adam {
                learning_rate,
                beta1,
                beta2,
                epsilon,
            } => write!(
                f,
                "adam(lr={learning_rate}, beta1={beta1}, beta2={beta2}, epsilon={epsilon})"
            ),
        }
    }
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
}
