//! Quality of the network on a dataset: loss, accuracy, regression errors.

use ndarray::{ArrayView1, ArrayView2, s};

use crate::config::Loss;
use crate::data::{Dataset, PreparedData, Scaler};
use crate::network::Mlp;

/// Number of samples evaluated at once. Evaluating a large set (MNIST) in
/// chunks keeps the intermediate activations small.
const CHUNK_ROWS: usize = 1024;

/// Quality of the network on one dataset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Metrics {
    /// Mean of the training loss (on normalized targets for regression).
    pub loss: f64,
    /// Fraction of correctly classified samples (classification only).
    pub accuracy: Option<f64>,
    /// Mean squared error in the original target scale (regression only).
    pub mse: Option<f64>,
    /// Mean absolute error in the original target scale (regression only).
    pub mae: Option<f64>,
}

impl Metrics {
    /// Evaluates `network` on `dataset`. A sample counts as correctly
    /// classified when its largest output is the one of its class.
    /// `target_scaler` brings regression errors back to the original scale.
    pub fn evaluate(
        network: &Mlp,
        dataset: &Dataset,
        loss: Loss,
        target_scaler: Option<&Scaler>,
    ) -> Self {
        let samples = dataset.len();
        assert!(samples > 0, "cannot evaluate on an empty dataset");
        let mut loss_sum = 0.0;
        let mut correct = 0;
        let mut squared_error = 0.0;
        let mut absolute_error = 0.0;

        for start in (0..samples).step_by(CHUNK_ROWS) {
            let end = (start + CHUNK_ROWS).min(samples);
            let inputs = dataset.inputs.slice(s![start..end, ..]);
            let targets = dataset.targets.slice(s![start..end, ..]).to_owned();
            let cache = network.forward(inputs);
            let output = cache
                .layers
                .last()
                .expect("a network has at least one layer");
            // The loss is a mean over the chunk; weighting it by the chunk
            // size gives the mean over the whole dataset in the end.
            loss_sum += loss.value(&output.z, &output.a, &targets) * (end - start) as f64;

            match &dataset.labels {
                Some(labels) => {
                    correct += count_correct(output.a.view(), labels.slice(s![start..end]));
                }
                None => {
                    // Normalization is (y - offset) / scale, so an error in the
                    // original scale is the normalized error times the scale.
                    let mut errors = &output.a - &targets;
                    if let Some(scaler) = target_scaler {
                        errors *= &scaler.scale;
                    }
                    squared_error += errors.iter().map(|error| error * error).sum::<f64>();
                    absolute_error += errors.iter().map(|error| error.abs()).sum::<f64>();
                }
            }
        }

        let loss = loss_sum / samples as f64;
        if dataset.labels.is_some() {
            Self {
                loss,
                accuracy: Some(correct as f64 / samples as f64),
                mse: None,
                mae: None,
            }
        } else {
            let elements = dataset.targets.len() as f64;
            Self {
                loss,
                accuracy: None,
                mse: Some(squared_error / elements),
                mae: Some(absolute_error / elements),
            }
        }
    }
}

/// Number of rows of `outputs` whose largest value is at the index in `labels`.
fn count_correct(outputs: ArrayView2<'_, f64>, labels: ArrayView1<'_, usize>) -> usize {
    outputs
        .rows()
        .into_iter()
        .zip(labels)
        .filter(|(row, label)| argmax(*row) == **label)
        .count()
}

/// Index of the largest value (the first one on ties).
fn argmax(row: ArrayView1<'_, f64>) -> usize {
    let mut best = 0;
    for (index, &value) in row.iter().enumerate() {
        if value > row[best] {
            best = index;
        }
    }
    best
}

/// Metrics of the network on every split of the data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitMetrics {
    pub train: Metrics,
    /// Present when part of the training file is held out for validation.
    pub validation: Option<Metrics>,
    /// Only observed: the test set never influences training.
    pub test: Metrics,
}

impl SplitMetrics {
    /// Evaluates `network` on the training, validation and test sets.
    pub fn evaluate(network: &Mlp, data: &PreparedData, loss: Loss) -> Self {
        let scaler = data.target_scaler.as_ref();
        let evaluate = |dataset: &Dataset| Metrics::evaluate(network, dataset, loss, scaler);
        Self {
            train: evaluate(&data.train),
            validation: data.validation.as_ref().map(evaluate),
            test: evaluate(&data.test),
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, array};

    use super::*;
    use crate::config::Normalization;
    use crate::network::{Activation, Layer};

    /// 2 -> 2 (identity) network that outputs its input.
    fn identity_network() -> Mlp {
        Mlp::from_layers(vec![Layer::from_parts(
            array![[1.0, 0.0], [0.0, 1.0]],
            None,
            Activation::Identity,
        )])
    }

    fn one_hot(labels: &Array1<usize>, classes: usize) -> Array2<f64> {
        Array2::from_shape_fn((labels.len(), classes), |(row, class)| {
            f64::from(u8::from(labels[row] == class))
        })
    }

    #[test]
    fn counts_largest_output_as_prediction() {
        let labels = array![0, 1, 0, 0];
        let dataset = Dataset {
            inputs: array![[1.0, 0.0], [0.0, 1.0], [2.0, 3.0], [5.0, 1.0]],
            targets: one_hot(&labels, 2),
            labels: Some(labels),
        };
        let metrics = Metrics::evaluate(&identity_network(), &dataset, Loss::Mse, None);
        // Predictions 0, 1, 1, 0: the third sample is wrong.
        assert_eq!(metrics.accuracy, Some(0.75));
        assert_eq!((metrics.mse, metrics.mae), (None, None));
        // Squared errors: 0, 0, 1 + 9, 16 + 1 over 4 x 2 outputs.
        assert_eq!(metrics.loss, 27.0 / 8.0);
    }

    #[test]
    fn argmax_takes_first_of_equal_values() {
        assert_eq!(argmax(array![1.0, 3.0, 3.0].view()), 1);
        assert_eq!(argmax(array![-1.0].view()), 0);
    }

    #[test]
    fn regression_errors_use_original_scale() {
        let network = Mlp::from_layers(vec![Layer::from_parts(
            array![[1.0]],
            Some(array![0.5]),
            Activation::Identity,
        )]);
        let dataset = Dataset {
            inputs: array![[0.0], [1.0]],
            targets: array![[0.0], [0.0]],
            labels: None,
        };
        let scaler = Scaler {
            normalization: Normalization::Standard,
            offset: array![100.0],
            scale: array![10.0],
        };
        let metrics = Metrics::evaluate(&network, &dataset, Loss::Mse, Some(&scaler));
        // Normalized errors 0.5 and 1.5 are 5 and 15 in the original scale.
        assert_eq!(metrics.loss, (0.25 + 2.25) / 2.0);
        assert_eq!(metrics.mse, Some((25.0 + 225.0) / 2.0));
        assert_eq!(metrics.mae, Some(10.0));
        assert_eq!(metrics.accuracy, None);
    }

    #[test]
    fn chunks_give_whole_dataset_mean() {
        let samples = 2 * CHUNK_ROWS + 37;
        let labels = Array1::from_shape_fn(samples, |row| row % 2);
        let inputs = Array2::from_shape_fn((samples, 2), |(row, column)| {
            ((row * 7 + column * 3) % 11) as f64 / 5.0 - 1.0
        });
        let dataset = Dataset {
            targets: one_hot(&labels, 2),
            inputs,
            labels: Some(labels),
        };
        let network = Mlp::from_layers(vec![Layer::from_parts(
            array![[0.3, -0.2], [0.1, 0.4]],
            Some(array![0.05, -0.05]),
            Activation::Softmax,
        )]);
        let metrics = Metrics::evaluate(&network, &dataset, Loss::CrossEntropy, None);
        let whole = network.loss(dataset.inputs.view(), &dataset.targets, Loss::CrossEntropy);
        assert!(
            (metrics.loss - whole).abs() < 1e-12,
            "{} != {whole}",
            metrics.loss
        );
        let predictions = network.predict(dataset.inputs.view());
        let correct = count_correct(predictions.view(), dataset.labels.as_ref().unwrap().view());
        assert_eq!(metrics.accuracy, Some(correct as f64 / samples as f64));
    }
}
