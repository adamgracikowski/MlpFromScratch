//! Top-level experiment configuration.

use serde::{Deserialize, Serialize};
use strum::Display;

use super::{DataConfig, LoggingConfig, ModelConfig, Optimizer, OutputConfig, TrainingConfig};

/// Full description of a single training run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentConfig {
    /// Kind of problem the network solves.
    pub task: Task,
    /// Dataset files and preprocessing.
    pub data: DataConfig,
    /// Network architecture and weight initialization.
    pub model: ModelConfig,
    /// Weight update rule and its hyperparameters.
    pub optimizer: Optimizer,
    /// Loss function, training length and RNG seed.
    pub training: TrainingConfig,
    /// How often training progress is recorded.
    pub logging: LoggingConfig,
    /// Settings of the prediction artifacts written after training.
    pub output: OutputConfig,
}

/// Kind of problem the network solves. MNIST is a classification task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Display)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum Task {
    Classification,
    Regression,
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::config::{HiddenActivation, Init, Loss, OutputActivation};

    /// A valid classification config used as the base for mutation tests.
    pub(crate) const CLASSIFICATION: &str = r#"
task = "classification"

[data]
train = "train.csv"
test = "test.csv"
target = "cls"
input_normalization = "standard"
target_normalization = "none"
validation_fraction = 0.0

[model]
hidden_layers = [16, 16]
bias = true
hidden_activation = { kind = "leaky_relu", alpha = 0.01 }
output_activation = "softmax"
init = { kind = "he" }

[optimizer]
kind = "adam"
learning_rate = 0.001

[training]
loss = "cross_entropy"
epochs = 200
batch_size = 32
seed = 42

[logging]
loss_every = 1
weights_every = 10
max_weights_per_layer = 64

[output]
grid_resolution = 200
grid_margin = 0.1
"#;

    fn parse(text: &str) -> Result<ExperimentConfig, toml::de::Error> {
        toml::from_str(text)
    }

    #[test]
    fn parses_full_config() {
        let config = parse(CLASSIFICATION).unwrap();
        assert_eq!(config.task, Task::Classification);
        assert_eq!(config.data.target, "cls");
        assert_eq!(config.model.hidden_layers, vec![16, 16]);
        assert_eq!(
            config.model.hidden_activation,
            HiddenActivation::LeakyRelu { alpha: 0.01 }
        );
        assert_eq!(config.model.output_activation, OutputActivation::Softmax);
        assert_eq!(config.model.init, Init::He);
        assert_eq!(config.training.loss, Loss::CrossEntropy);
        assert_eq!(config.training.seed, 42);
    }

    #[test]
    fn rejects_unknown_field() {
        let text = CLASSIFICATION.replace("epochs = 200", "epoch = 200");
        assert!(parse(&text).unwrap_err().message().contains("epoch"));
    }

    #[test]
    fn rejects_missing_field() {
        let text = CLASSIFICATION.replace("seed = 42", "");
        assert!(parse(&text).unwrap_err().message().contains("seed"));
    }

    #[test]
    fn rejects_unknown_section() {
        let text = format!("{CLASSIFICATION}\n[extra]\nvalue = 1\n");
        assert!(parse(&text).is_err());
    }

    #[test]
    fn round_trips_through_toml() {
        let config = parse(CLASSIFICATION).unwrap();
        let text = toml::to_string(&config).unwrap();
        assert_eq!(parse(&text).unwrap(), config);
    }

    #[test]
    fn displays_task_like_config() {
        assert_eq!(Task::Classification.to_string(), "classification");
        assert_eq!(Task::Regression.to_string(), "regression");
    }
}
