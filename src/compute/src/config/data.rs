//! Dataset files and preprocessing.

use std::fmt;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Dataset files and preprocessing.
///
/// Paths are resolved relative to the current working directory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataConfig {
    /// CSV file with the training set.
    pub train: PathBuf,
    /// CSV file with the test set, used only for evaluation.
    pub test: PathBuf,
    /// Name of the target column; all other columns are input features.
    pub target: String,
    /// Normalization of the input features.
    pub input_normalization: Normalization,
    /// Normalization of the target (regression only).
    pub target_normalization: Normalization,
    /// Number of training samples drawn at random; `None` uses the whole set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub train_subset: Option<usize>,
    /// Fraction of the training set held out for validation (0 disables it).
    pub validation_fraction: f64,
}

/// Feature or target normalization. Statistics come from the training set only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Normalization {
    /// Leave values unchanged.
    None,
    /// Shift to zero mean and scale to unit variance.
    Standard,
    /// Rescale linearly to the `[0, 1]` range.
    MinMax,
}

impl fmt::Display for Normalization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Normalization::None => "none",
            Normalization::Standard => "standard",
            Normalization::MinMax => "min_max",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DATA: &str = r#"
train = "train.csv"
test = "test.csv"
target = "cls"
input_normalization = "min_max"
target_normalization = "none"
validation_fraction = 0.2
"#;

    #[test]
    fn train_subset_is_optional() {
        let data: DataConfig = toml::from_str(DATA).unwrap();
        assert_eq!(data.train_subset, None);
        assert_eq!(data.input_normalization, Normalization::MinMax);
    }

    #[test]
    fn parses_train_subset() {
        let text = format!("{DATA}train_subset = 100\n");
        let data: DataConfig = toml::from_str(&text).unwrap();
        assert_eq!(data.train_subset, Some(100));
    }
}
