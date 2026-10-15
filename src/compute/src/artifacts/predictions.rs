//! Predictions of the trained network on the training, validation and test sets.

use std::path::Path;

use ndarray::{Array1, Array2, ArrayView2, Axis, s};

use super::npy::write_npy;
use super::{ArtifactError, create_dir};
use crate::data::{Dataset, PreparedData};
use crate::network::Mlp;
use crate::training::{CHUNK_ROWS, argmax};

/// Network output for every row of `inputs` (`n x d`), computed in chunks
/// of [`CHUNK_ROWS`] rows to keep the intermediate activations small.
pub fn predict_in_chunks(network: &Mlp, inputs: ArrayView2<'_, f64>) -> Array2<f64> {
    let samples = inputs.nrows();
    let mut outputs = Array2::zeros((samples, network.output_size()));
    for start in (0..samples).step_by(CHUNK_ROWS) {
        let end = (start + CHUNK_ROWS).min(samples);
        let chunk = network.predict(inputs.slice(s![start..end, ..]));
        outputs.slice_mut(s![start..end, ..]).assign(&chunk);
    }
    outputs
}

/// True and predicted targets of one dataset, in the terms of the source
/// file: original class labels or target values in the original scale.
#[derive(Debug, Clone, PartialEq)]
pub enum Targets {
    Classification {
        /// True label of every sample (`n`).
        labels: Array1<i64>,
        /// Label of the largest output of every sample (`n`).
        predicted: Array1<i64>,
    },
    Regression {
        /// True target values (`n x 1`).
        targets: Array2<f64>,
    },
}

/// Predictions of the network on one dataset.
#[derive(Debug, Clone, PartialEq)]
pub struct Predictions {
    /// Row of every sample in its source file (`n`).
    pub rows: Array1<usize>,
    /// Network output (`n x K`): class probabilities for classification,
    /// predicted values in the original target scale for regression.
    pub outputs: Array2<f64>,
    pub targets: Targets,
}

impl Predictions {
    /// Writes to `dir`: `rows.npy` (`u64`), `outputs.npy` (`n x K`) and
    /// either `labels.npy` and `predicted.npy` (`i64`, classification) or
    /// `targets.npy` (`n x 1`, regression).
    pub(super) fn write(&self, dir: &Path) -> Result<(), ArtifactError> {
        create_dir(dir)?;
        let samples = self.rows.len();
        let rows = self.rows.iter().map(|&row| row as u64);
        write_npy(&dir.join("rows.npy"), &[samples], rows)?;
        let shape = self.outputs.shape();
        write_npy(
            &dir.join("outputs.npy"),
            shape,
            self.outputs.iter().copied(),
        )?;
        match &self.targets {
            Targets::Classification { labels, predicted } => {
                write_npy(&dir.join("labels.npy"), &[samples], labels.iter().copied())?;
                write_npy(
                    &dir.join("predicted.npy"),
                    &[samples],
                    predicted.iter().copied(),
                )
            }
            Targets::Regression { targets } => write_npy(
                &dir.join("targets.npy"),
                targets.shape(),
                targets.iter().copied(),
            ),
        }
    }

    /// Runs `network` on `dataset`; `data` provides the class labels and the
    /// target normalization to undo.
    pub fn compute(network: &Mlp, dataset: &Dataset, data: &PreparedData) -> Self {
        let mut outputs = predict_in_chunks(network, dataset.inputs.view());
        let targets = match (&data.classes, &dataset.labels) {
            (Some(classes), Some(labels)) => {
                let classes = classes.classes();
                Targets::Classification {
                    labels: labels.mapv(|label| classes[label]),
                    predicted: outputs
                        .axis_iter(Axis(0))
                        .map(|row| classes[argmax(row)])
                        .collect(),
                }
            }
            _ => {
                let mut targets = dataset.targets.clone();
                if let Some(scaler) = &data.target_scaler {
                    outputs = scaler.inverse_transform(&outputs);
                    targets = scaler.inverse_transform(&targets);
                }
                Targets::Regression { targets }
            }
        };
        Self {
            rows: dataset.rows.clone(),
            outputs,
            targets,
        }
    }
}

/// Predictions on every split of the data.
#[derive(Debug, Clone, PartialEq)]
pub struct SplitPredictions {
    pub train: Predictions,
    /// Present when part of the training file is held out for validation.
    pub validation: Option<Predictions>,
    pub test: Predictions,
}

impl SplitPredictions {
    /// Writes the predictions of every split to `dir/<split>/`.
    pub(super) fn write(&self, dir: &Path) -> Result<(), ArtifactError> {
        self.train.write(&dir.join("train"))?;
        if let Some(validation) = &self.validation {
            validation.write(&dir.join("validation"))?;
        }
        self.test.write(&dir.join("test"))
    }

    /// Runs `network` on the training, validation and test sets.
    pub fn compute(network: &Mlp, data: &PreparedData) -> Self {
        let compute = |dataset: &Dataset| Predictions::compute(network, dataset, data);
        Self {
            train: compute(&data.train),
            validation: data.validation.as_ref().map(compute),
            test: compute(&data.test),
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;
    use crate::config::Normalization;
    use crate::data::{ClassEncoding, Scaler, Table};
    use crate::network::{Activation, Layer};

    /// 1 -> 1 network computing `2x + 1`.
    fn linear() -> Mlp {
        Mlp::from_layers(vec![Layer::from_parts(
            array![[2.0]],
            Some(array![1.0]),
            Activation::Identity,
        )])
    }

    fn regression_data(dataset: Dataset, target_scaler: Scaler) -> PreparedData {
        PreparedData {
            input_scaler: Scaler::fit(Normalization::None, &dataset.inputs),
            target_scaler: Some(target_scaler),
            validation: None,
            test: dataset.clone(),
            train: dataset,
            feature_names: vec!["x".to_owned()],
            classes: None,
        }
    }

    #[test]
    fn chunks_give_same_output_as_whole_batch() {
        let network = linear();
        let inputs = Array2::from_shape_fn((2 * CHUNK_ROWS + 3, 1), |(row, _)| row as f64);
        assert_eq!(
            predict_in_chunks(&network, inputs.view()),
            network.predict(inputs.view())
        );
    }

    #[test]
    fn regression_predictions_use_original_scale() {
        let mut dataset = Dataset::new(
            array![[0.0], [1.0], [2.0]],
            array![[1.0], [2.0], [3.0]],
            None,
        );
        dataset.rows = array![7, 3, 5];
        // Targets normalized as (y - 10) / 4.
        let scaler = Scaler {
            normalization: Normalization::Standard,
            offset: array![10.0],
            scale: array![4.0],
        };
        let data = regression_data(dataset.clone(), scaler);
        let predictions = Predictions::compute(&linear(), &dataset, &data);

        assert_eq!(predictions.rows, array![7, 3, 5]);
        // Network outputs 1, 3, 5 map to 14, 22, 30.
        assert_eq!(predictions.outputs, array![[14.0], [22.0], [30.0]]);
        let Targets::Regression { targets } = predictions.targets else {
            panic!("expected regression targets");
        };
        assert_eq!(targets, array![[14.0], [18.0], [22.0]]);
    }

    #[test]
    fn classification_predictions_use_original_labels() {
        // The network outputs its input. Classes 2 and 5 of the source file.
        let network = Mlp::from_layers(vec![Layer::from_parts(
            array![[1.0, 0.0], [0.0, 1.0]],
            None,
            Activation::Identity,
        )]);
        let labels = array![0, 1, 1];
        let targets = labels.mapv(|label| f64::from(u8::from(label == 1)));
        let dataset = Dataset::new(
            array![[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]],
            Array2::from_shape_fn((3, 2), |(row, class)| {
                if class == 1 {
                    targets[row]
                } else {
                    1.0 - targets[row]
                }
            }),
            Some(labels),
        );
        let table = Table {
            path: "train.csv".into(),
            feature_names: vec!["x".to_owned(), "y".to_owned()],
            features: Array2::zeros((2, 2)),
            target: vec![5.0, 2.0],
            lines: vec![2, 3],
        };
        let data = PreparedData {
            input_scaler: Scaler::fit(Normalization::None, &dataset.inputs),
            target_scaler: None,
            validation: None,
            test: dataset.clone(),
            train: dataset.clone(),
            feature_names: vec!["x".to_owned(), "y".to_owned()],
            classes: Some(ClassEncoding::fit(&table).unwrap()),
        };
        let predictions = Predictions::compute(&network, &dataset, &data);

        assert_eq!(predictions.outputs, dataset.inputs);
        assert_eq!(
            predictions.targets,
            Targets::Classification {
                labels: array![2, 5, 5],
                predicted: array![2, 5, 2],
            }
        );
    }

    #[test]
    fn computes_every_split() {
        let dataset = Dataset::new(array![[0.0]], array![[1.0]], None);
        let mut data = regression_data(
            dataset.clone(),
            Scaler::fit(Normalization::None, &dataset.targets),
        );
        data.validation = Some(dataset);
        let predictions = SplitPredictions::compute(&linear(), &data);
        assert!(predictions.validation.is_some());
        assert_eq!(predictions.test.outputs, array![[1.0]]);
    }
}
