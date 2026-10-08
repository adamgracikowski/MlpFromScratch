//! Loss function, training length and RNG seed.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Loss function, training length and RNG seed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingConfig {
    /// Loss minimized during training.
    pub loss: Loss,
    /// Number of passes over the training set.
    pub epochs: usize,
    /// Number of samples per gradient step.
    pub batch_size: usize,
    /// Seed of every random choice in the run.
    pub seed: u64,
}

/// Loss function minimized during training.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Loss {
    /// Categorical cross-entropy (classification, softmax output).
    CrossEntropy,
    /// Mean squared error (regression, or classification on one-hot targets).
    Mse,
    /// Mean absolute error (regression).
    Mae,
}

impl fmt::Display for Loss {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Loss::CrossEntropy => "cross_entropy",
            Loss::Mse => "mse",
            Loss::Mae => "mae",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_negative_epochs() {
        let text = "loss = \"mae\"\nepochs = -1\nbatch_size = 32\nseed = 1\n";
        assert!(toml::from_str::<TrainingConfig>(text).is_err());
    }
}
