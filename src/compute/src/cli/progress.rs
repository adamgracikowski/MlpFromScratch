//! Training progress printed to the console.

use std::fmt;

use crate::network::Mlp;
use crate::training::{EpochRecord, Metrics, SplitMetrics, TrainingObserver};

/// Number of progress lines printed by default.
const DEFAULT_LINES: usize = 20;

/// Prints the metrics before training, every `every` epochs and after the
/// last epoch; `every = 0` prints nothing.
pub(super) struct Progress {
    every: usize,
    epochs: usize,
}

impl Progress {
    /// Progress of a run of `epochs` epochs; `every` defaults to about
    /// [`DEFAULT_LINES`] lines per run.
    pub(super) fn new(every: Option<usize>, epochs: usize) -> Self {
        let every = every.unwrap_or_else(|| (epochs / DEFAULT_LINES).max(1));
        Self { every, epochs }
    }

    fn prints(&self, epoch: usize) -> bool {
        self.every > 0 && (epoch % self.every == 0 || epoch == self.epochs)
    }
}

impl TrainingObserver for Progress {
    fn on_epoch(&mut self, record: &EpochRecord, _network: &Mlp) {
        if self.prints(record.epoch) {
            println!("{}", EpochLine(record, self.epochs));
        }
    }
}

/// One progress line, e.g.
/// `epoch  20/200  train loss 0.1234 acc 95.20% | test loss 0.1300 acc 94.00%`.
struct EpochLine<'a>(&'a EpochRecord, usize);

impl fmt::Display for EpochLine<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let EpochLine(record, epochs) = *self;
        let width = epochs.to_string().len();
        write!(
            f,
            "epoch {:>width$}/{epochs}  {}",
            record.epoch,
            SplitLine(&record.metrics)
        )
    }
}

/// Metrics of every split, e.g. `train loss 0.1234 acc 95.20% | test ...`.
pub(super) struct SplitLine<'a>(pub &'a SplitMetrics);

impl fmt::Display for SplitLine<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let metrics = self.0;
        write!(f, "train {}", MetricsLine(&metrics.train))?;
        if let Some(validation) = &metrics.validation {
            write!(f, " | val {}", MetricsLine(validation))?;
        }
        write!(f, " | test {}", MetricsLine(&metrics.test))
    }
}

/// Metrics of one split: loss, then accuracy or the regression errors in
/// the original target scale.
struct MetricsLine<'a>(&'a Metrics);

impl fmt::Display for MetricsLine<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let metrics = self.0;
        write!(f, "loss {}", Number(metrics.loss))?;
        if let Some(accuracy) = metrics.accuracy {
            write!(f, " acc {:.2}%", 100.0 * accuracy)?;
        }
        if let Some(mse) = metrics.mse {
            write!(f, " mse {}", Number(mse))?;
        }
        if let Some(mae) = metrics.mae {
            write!(f, " mae {}", Number(mae))?;
        }
        Ok(())
    }
}

/// A metric value with 4 decimals, or in scientific notation when it is very
/// large or very small (e.g. a diverged loss), so that lines stay short.
struct Number(f64);

impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = self.0;
        let magnitude = value.abs();
        if magnitude >= 1e6 || (magnitude > 0.0 && magnitude < 1e-3) {
            write!(f, "{value:.4e}")
        } else {
            write!(f, "{value:.4}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classification(loss: f64, accuracy: f64) -> Metrics {
        Metrics {
            loss,
            accuracy: Some(accuracy),
            mse: None,
            mae: None,
        }
    }

    #[test]
    fn default_interval_gives_about_twenty_lines() {
        assert_eq!(Progress::new(None, 200).every, 10);
        assert_eq!(Progress::new(None, 5).every, 1);
        assert_eq!(Progress::new(Some(7), 200).every, 7);
    }

    #[test]
    fn prints_first_selected_and_last_epochs() {
        let progress = Progress::new(Some(4), 10);
        let printed: Vec<usize> = (0..=10).filter(|&epoch| progress.prints(epoch)).collect();
        assert_eq!(printed, [0, 4, 8, 10]);
        let silent = Progress::new(Some(0), 10);
        assert!((0..=10).all(|epoch| !silent.prints(epoch)));
    }

    #[test]
    fn formats_classification_epoch() {
        let record = EpochRecord {
            epoch: 20,
            iteration: 320,
            metrics: SplitMetrics {
                train: classification(0.12344, 0.9523),
                validation: Some(classification(0.2, 0.9)),
                test: classification(0.13, 0.94),
            },
        };
        assert_eq!(
            EpochLine(&record, 200).to_string(),
            "epoch  20/200  train loss 0.1234 acc 95.23% \
             | val loss 0.2000 acc 90.00% | test loss 0.1300 acc 94.00%"
        );
    }

    #[test]
    fn formats_regression_errors() {
        let metrics = Metrics {
            loss: 0.5,
            accuracy: None,
            mse: Some(125.0),
            mae: Some(10.0),
        };
        assert_eq!(
            MetricsLine(&metrics).to_string(),
            "loss 0.5000 mse 125.0000 mae 10.0000"
        );
    }

    #[test]
    fn formats_extreme_numbers_in_scientific_notation() {
        let numbers = [
            0.0,
            0.0012,
            0.000123,
            123456.0,
            7.6e300,
            f64::INFINITY,
            f64::NAN,
        ];
        let texts = numbers.map(|value| Number(value).to_string());
        assert_eq!(
            texts,
            [
                "0.0000",
                "0.0012",
                "1.2300e-4",
                "123456.0000",
                "7.6000e300",
                "inf",
                "NaN"
            ]
        );
    }
}
