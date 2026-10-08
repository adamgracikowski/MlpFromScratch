//! Network architecture, activations and weight initialization.

use std::fmt;

use serde::{Deserialize, Serialize};

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
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum HiddenActivation {
    Sigmoid,
    Tanh,
    Relu,
    /// ReLU with slope `alpha` for negative inputs.
    LeakyRelu {
        alpha: f64,
    },
}

/// Activation function of the output layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputActivation {
    Identity,
    Sigmoid,
    Softmax,
}

/// Weight initialization scheme. Biases always start at zero.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Init {
    /// Glorot/Xavier uniform, suited to sigmoid and tanh.
    Xavier,
    /// He/Kaiming normal, suited to ReLU and leaky ReLU.
    He,
    /// Uniform on `[-limit, limit]`.
    Uniform { limit: f64 },
}

impl fmt::Display for HiddenActivation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HiddenActivation::Sigmoid => f.write_str("sigmoid"),
            HiddenActivation::Tanh => f.write_str("tanh"),
            HiddenActivation::Relu => f.write_str("relu"),
            HiddenActivation::LeakyRelu { alpha } => write!(f, "leaky_relu(alpha={alpha})"),
        }
    }
}

impl fmt::Display for OutputActivation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            OutputActivation::Identity => "identity",
            OutputActivation::Sigmoid => "sigmoid",
            OutputActivation::Softmax => "softmax",
        })
    }
}

impl fmt::Display for Init {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Init::Xavier => f.write_str("xavier"),
            Init::He => f.write_str("he"),
            Init::Uniform { limit } => write!(f, "uniform(limit={limit})"),
        }
    }
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
}
