//! Metrics of every epoch as a CSV table.

use std::path::Path;

use super::ArtifactError;
use crate::training::{EpochRecord, Metrics};

/// Writes one row per epoch record: `epoch`, `iteration`, then for every
/// split (`train`, `validation` when present, `test`) its loss and either
/// the accuracy (classification) or the MSE and MAE in the original scale
/// (regression), e.g. `train_loss,train_accuracy,test_loss,test_accuracy`.
pub(super) fn write_epochs_csv(path: &Path, epochs: &[EpochRecord]) -> Result<(), ArtifactError> {
    let error = |source: csv::Error| ArtifactError::write(path, source);
    let mut writer = csv::Writer::from_path(path).map_err(error)?;
    if let Some(first) = epochs.first() {
        writer.write_record(header(first)).map_err(error)?;
    }
    for record in epochs {
        writer.write_record(row(record)).map_err(error)?;
    }
    writer
        .flush()
        .map_err(|source| ArtifactError::write(path, source))
}

/// The metrics of every split present in `record`, with the split names.
fn splits(record: &EpochRecord) -> Vec<(&'static str, &Metrics)> {
    let metrics = &record.metrics;
    let mut splits = vec![("train", &metrics.train)];
    if let Some(validation) = &metrics.validation {
        splits.push(("validation", validation));
    }
    splits.push(("test", &metrics.test));
    splits
}

/// Names and values of the metrics present in `metrics`.
fn values(metrics: &Metrics) -> Vec<(&'static str, f64)> {
    [
        ("loss", Some(metrics.loss)),
        ("accuracy", metrics.accuracy),
        ("mse", metrics.mse),
        ("mae", metrics.mae),
    ]
    .into_iter()
    .filter_map(|(name, value)| value.map(|value| (name, value)))
    .collect()
}

fn header(record: &EpochRecord) -> Vec<String> {
    let mut header = vec!["epoch".to_owned(), "iteration".to_owned()];
    for (split, metrics) in splits(record) {
        header.extend(
            values(metrics)
                .into_iter()
                .map(|(name, _)| format!("{split}_{name}")),
        );
    }
    header
}

fn row(record: &EpochRecord) -> Vec<String> {
    let mut row = vec![record.epoch.to_string(), record.iteration.to_string()];
    for (_, metrics) in splits(record) {
        row.extend(
            values(metrics)
                .into_iter()
                .map(|(_, value)| value.to_string()),
        );
    }
    row
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::training::SplitMetrics;

    fn classification(loss: f64) -> Metrics {
        Metrics {
            loss,
            accuracy: Some(0.5),
            mse: None,
            mae: None,
        }
    }

    fn write(epochs: &[EpochRecord]) -> String {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("epochs.csv");
        write_epochs_csv(&path, epochs).unwrap();
        std::fs::read_to_string(path).unwrap()
    }

    #[test]
    fn writes_classification_metrics_of_every_split() {
        let record = |epoch, loss| EpochRecord {
            epoch,
            iteration: 10 * epoch,
            metrics: SplitMetrics {
                train: classification(loss),
                validation: Some(classification(0.25)),
                test: classification(f64::NAN),
            },
        };
        assert_eq!(
            write(&[record(0, 1.5), record(1, 0.75)]),
            "epoch,iteration,train_loss,train_accuracy,validation_loss,validation_accuracy,\
             test_loss,test_accuracy\n\
             0,0,1.5,0.5,0.25,0.5,NaN,0.5\n\
             1,10,0.75,0.5,0.25,0.5,NaN,0.5\n"
        );
    }

    #[test]
    fn writes_regression_metrics_without_validation() {
        let metrics = Metrics {
            loss: 0.1,
            accuracy: None,
            mse: Some(4.0),
            mae: Some(2.0),
        };
        let record = EpochRecord {
            epoch: 0,
            iteration: 0,
            metrics: SplitMetrics {
                train: metrics,
                validation: None,
                test: metrics,
            },
        };
        assert_eq!(
            write(&[record]),
            "epoch,iteration,train_loss,train_mse,train_mae,test_loss,test_mse,test_mae\n\
             0,0,0.1,4,2,0.1,4,2\n"
        );
    }
}
