//! Loss function, training length, RNG seed and early stopping.

use serde::{Deserialize, Serialize};
use strum::Display;

/// Loss function, training length, RNG seed and early stopping.
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
    /// Stop when the validation loss stops improving; `None` runs every epoch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub early_stopping: Option<EarlyStopping>,
}

/// Stops training once the validation loss has not improved for `patience`
/// epochs, then restores the weights of the best epoch.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EarlyStopping {
    /// Number of epochs without improvement after which training stops.
    pub patience: usize,
    /// Smallest decrease of the validation loss that counts as an improvement.
    #[serde(default)]
    pub min_delta: f64,
}

/// Loss function minimized during training.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Display)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum Loss {
    /// Categorical cross-entropy (classification, softmax output).
    CrossEntropy,
    /// Mean squared error (regression, or classification on one-hot targets).
    Mse,
    /// Mean absolute error (regression).
    Mae,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_negative_epochs() {
        let text = "loss = \"mae\"\nepochs = -1\nbatch_size = 32\nseed = 1\n";
        assert!(toml::from_str::<TrainingConfig>(text).is_err());
    }

    const TRAINING: &str = "loss = \"mse\"\nepochs = 10\nbatch_size = 32\nseed = 1\n";

    #[test]
    fn early_stopping_is_optional() {
        let training: TrainingConfig = toml::from_str(TRAINING).unwrap();
        assert_eq!(training.early_stopping, None);
    }

    #[test]
    fn min_delta_defaults_to_zero() {
        let text = format!("{TRAINING}early_stopping = {{ patience = 5 }}\n");
        let training: TrainingConfig = toml::from_str(&text).unwrap();
        let expected = EarlyStopping {
            patience: 5,
            min_delta: 0.0,
        };
        assert_eq!(training.early_stopping, Some(expected));
    }

    #[test]
    fn parses_early_stopping_with_min_delta() {
        let text = format!("{TRAINING}early_stopping = {{ patience = 5, min_delta = 1e-4 }}\n");
        let training: TrainingConfig = toml::from_str(&text).unwrap();
        assert_eq!(training.early_stopping.unwrap().min_delta, 1e-4);
    }

    #[test]
    fn rejects_unknown_early_stopping_field() {
        let text = format!("{TRAINING}early_stopping = {{ patience = 5, monitor = \"loss\" }}\n");
        assert!(toml::from_str::<TrainingConfig>(&text).is_err());
    }

    #[test]
    fn displays_loss_like_config() {
        let names = [Loss::CrossEntropy, Loss::Mse, Loss::Mae].map(|loss| loss.to_string());
        assert_eq!(names, ["cross_entropy", "mse", "mae"]);
    }
}
