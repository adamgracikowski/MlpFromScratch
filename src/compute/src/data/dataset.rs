//! Datasets in the form consumed by the network.

use ndarray::{Array1, Array2, Axis};

use super::{ClassEncoding, Scaler};

/// Samples of one split (train, validation or test), one row per sample.
#[derive(Debug, Clone, PartialEq)]
pub struct Dataset {
    /// Input features (`n x d`), normalized once the data is prepared.
    pub inputs: Array2<f64>,
    /// Network targets: one-hot classes (`n x K`) for classification,
    /// target values (`n x 1`, normalized once prepared) for regression.
    pub targets: Array2<f64>,
    /// Class index of every sample (`n`), for classification only.
    pub labels: Option<Array1<usize>>,
}

impl Dataset {
    /// Number of samples.
    pub fn len(&self) -> usize {
        self.inputs.nrows()
    }

    /// Whether the dataset has no samples.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The samples at `rows`, in the given order.
    pub fn select(&self, rows: &[usize]) -> Dataset {
        Dataset {
            inputs: self.inputs.select(Axis(0), rows),
            targets: self.targets.select(Axis(0), rows),
            labels: self
                .labels
                .as_ref()
                .map(|labels| labels.select(Axis(0), rows)),
        }
    }

    /// Number of samples of every class, for classification only.
    pub fn class_counts(&self, class_count: usize) -> Option<Vec<usize>> {
        let labels = self.labels.as_ref()?;
        let mut counts = vec![0; class_count];
        for &label in labels {
            counts[label] += 1;
        }
        Some(counts)
    }

    /// Normalizes inputs and, if given, targets in place.
    pub(super) fn normalize(&mut self, inputs: &Scaler, targets: Option<&Scaler>) {
        inputs.transform_in_place(&mut self.inputs);
        if let Some(targets) = targets {
            targets.transform_in_place(&mut self.targets);
        }
    }
}

/// Training, validation and test sets ready for training, with everything
/// needed to interpret the network output.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedData {
    /// Samples the network is trained on.
    pub train: Dataset,
    /// Samples held out from the training file (`validation_fraction > 0`).
    pub validation: Option<Dataset>,
    /// Samples used only for evaluation.
    pub test: Dataset,
    /// Names of the input features, in file order.
    pub feature_names: Vec<String>,
    /// Normalization of the inputs, fitted on `train`.
    pub input_scaler: Scaler,
    /// Normalization of the target, fitted on `train` (regression only).
    /// Its inverse brings predictions back to the original scale.
    pub target_scaler: Option<Scaler>,
    /// Class labels behind the output indices (classification only).
    pub classes: Option<ClassEncoding>,
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    fn dataset() -> Dataset {
        Dataset {
            inputs: array![[0.0], [1.0], [2.0]],
            targets: array![[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            labels: Some(array![0, 1, 1]),
        }
    }

    #[test]
    fn selects_rows() {
        let selected = dataset().select(&[2, 0]);
        assert_eq!(selected.inputs, array![[2.0], [0.0]]);
        assert_eq!(selected.targets, array![[0.0, 1.0], [1.0, 0.0]]);
        assert_eq!(selected.labels, Some(array![1, 0]));
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn counts_classes() {
        assert_eq!(dataset().class_counts(3), Some(vec![1, 2, 0]));
        let mut regression = dataset();
        regression.labels = None;
        assert_eq!(regression.class_counts(2), None);
    }
}
