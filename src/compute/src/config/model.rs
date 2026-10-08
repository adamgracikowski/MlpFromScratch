//! Network architecture, activations and weight initialization.

use serde::{Deserialize, Serialize};
use strum::Display;

/// Network architecture and weight initialization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// Sizes of the hidden layers, from input to output (0 to 4 layers).
    pub hidden_layers: Vec<usize>,
    /// Whether every layer has a bias vector.
    pub bias: bool,
    /// Activation of all hidden layers.
    pub hidden_activation: HiddenActivation,
    /// Activation of the output layer; must match the task and loss.
    pub output_activation: OutputActivation,
    /// Initialization scheme of the weights.
    pub init: Init,
}

/// Activation function of the hidden layers.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Display)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
#[strum(serialize_all = "snake_case")]
pub enum HiddenActivation {
    Sigmoid,
    Tanh,
    Relu,
    /// ReLU with slope `alpha` for negative inputs.
    #[strum(to_string = "leaky_relu(alpha={alpha})")]
    LeakyRelu {
        alpha: f64,
    },
}

/// Activation function of the output layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Display)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum OutputActivation {
    Identity,
    Sigmoid,
    Softmax,
}

/// Weight initialization scheme. Biases always start at zero.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Display)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
#[strum(serialize_all = "snake_case")]
pub enum Init {
    /// Glorot/Xavier uniform, suited to sigmoid and tanh.
    Xavier,
    /// He/Kaiming normal, suited to ReLU and leaky ReLU.
    He,
    /// Uniform on `[-limit, limit]`.
    #[strum(to_string = "uniform(limit={limit})")]
    Uniform { limit: f64 },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(hidden_activation: &str, output_activation: &str, init: &str) -> String {
        format!(
            "hidden_layers = []\n\
             bias = false\n\
             hidden_activation = {hidden_activation}\n\
             output_activation = {output_activation}\n\
             init = {init}\n"
        )
    }

    fn parse(text: &str) -> Result<ModelConfig, toml::de::Error> {
        toml::from_str(text)
    }

    #[test]
    fn parses_parameterless_tagged_variants() {
        let text = model(
            r#"{ kind = "sigmoid" }"#,
            r#""identity""#,
            r#"{ kind = "xavier" }"#,
        );
        let model = parse(&text).unwrap();
        assert_eq!(model.hidden_activation, HiddenActivation::Sigmoid);
        assert_eq!(model.init, Init::Xavier);
        assert!(model.hidden_layers.is_empty());
    }

    #[test]
    fn parses_tagged_variants_with_parameters() {
        let text = model(
            r#"{ kind = "leaky_relu", alpha = 0.05 }"#,
            r#""softmax""#,
            r#"{ kind = "uniform", limit = 0.5 }"#,
        );
        let model = parse(&text).unwrap();
        assert_eq!(
            model.hidden_activation,
            HiddenActivation::LeakyRelu { alpha: 0.05 }
        );
        assert_eq!(model.init, Init::Uniform { limit: 0.5 });
    }

    #[test]
    fn rejects_missing_variant_parameter() {
        let text = model(
            r#"{ kind = "leaky_relu" }"#,
            r#""identity""#,
            r#"{ kind = "he" }"#,
        );
        assert!(parse(&text).is_err());
    }

    #[test]
    fn rejects_unknown_output_activation() {
        let text = model(
            r#"{ kind = "tanh" }"#,
            r#""softplus""#,
            r#"{ kind = "he" }"#,
        );
        assert!(parse(&text).is_err());
    }

    #[test]
    fn displays_activations_and_init() {
        let hidden = [
            HiddenActivation::Sigmoid,
            HiddenActivation::Tanh,
            HiddenActivation::Relu,
            HiddenActivation::LeakyRelu { alpha: 0.01 },
        ]
        .map(|activation| activation.to_string());
        assert_eq!(
            hidden,
            ["sigmoid", "tanh", "relu", "leaky_relu(alpha=0.01)"]
        );
        let output = [
            OutputActivation::Identity,
            OutputActivation::Sigmoid,
            OutputActivation::Softmax,
        ]
        .map(|activation| activation.to_string());
        assert_eq!(output, ["identity", "sigmoid", "softmax"]);
        let init =
            [Init::Xavier, Init::He, Init::Uniform { limit: 0.5 }].map(|init| init.to_string());
        assert_eq!(init, ["xavier", "he", "uniform(limit=0.5)"]);
    }
}
