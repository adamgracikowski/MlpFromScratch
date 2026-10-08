//! Checks of the `[optimizer]` section.

use crate::config::{Issue, Optimizer};

/// Checks the learning rate and the parameters of the chosen optimizer.
pub(super) fn validate_optimizer(optimizer: &Optimizer, issues: &mut Vec<Issue>) {
    let learning_rate = match *optimizer {
        Optimizer::Sgd { learning_rate } => learning_rate,
        Optimizer::Momentum {
            learning_rate,
            momentum,
        } => {
            check_unit_interval(momentum, &["optimizer", "momentum"], issues);
            learning_rate
        }
        Optimizer::Adam {
            learning_rate,
            beta1,
            beta2,
            epsilon,
        } => {
            check_unit_interval(beta1, &["optimizer", "beta1"], issues);
            check_unit_interval(beta2, &["optimizer", "beta2"], issues);
            if !(epsilon > 0.0 && epsilon.is_finite()) {
                issues.push(Issue::error(&["optimizer", "epsilon"], "must be positive"));
            }
            learning_rate
        }
    };
    if !(learning_rate > 0.0 && learning_rate.is_finite()) {
        issues.push(Issue::error(
            &["optimizer", "learning_rate"],
            "must be positive",
        ));
    }
}

/// Pushes an error unless `value` lies in `[0, 1)`.
fn check_unit_interval(value: f64, path: &[&'static str], issues: &mut Vec<Issue>) {
    if !(0.0..1.0).contains(&value) {
        issues.push(Issue::error(path, "must be in [0, 1)"));
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixtures::{classification, errors};
    use super::*;

    #[test]
    fn rejects_non_positive_learning_rate() {
        for learning_rate in [0.0, -0.1, f64::INFINITY] {
            let mut config = classification();
            config.optimizer = Optimizer::Sgd { learning_rate };
            assert_eq!(errors(&config), ["optimizer.learning_rate"]);
        }
    }

    #[test]
    fn rejects_momentum_out_of_range() {
        let mut config = classification();
        config.optimizer = Optimizer::Momentum {
            learning_rate: 0.1,
            momentum: 1.0,
        };
        assert_eq!(errors(&config), ["optimizer.momentum"]);
    }

    #[test]
    fn rejects_adam_parameters_out_of_range() {
        let mut config = classification();
        config.optimizer = Optimizer::Adam {
            learning_rate: 0.001,
            beta1: 1.0,
            beta2: -0.1,
            epsilon: 0.0,
        };
        assert_eq!(
            errors(&config),
            ["optimizer.beta1", "optimizer.beta2", "optimizer.epsilon"]
        );
    }
}
