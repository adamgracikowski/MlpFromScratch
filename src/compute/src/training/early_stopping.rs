//! Early stopping: ends training once the validation loss stops improving
//! and keeps the network of the best epoch.

use super::SplitMetrics;
use crate::config::EarlyStopping;
use crate::network::Mlp;

/// The network of one epoch together with its metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct Checkpoint {
    pub epoch: usize,
    pub metrics: SplitMetrics,
    pub network: Mlp,
}

/// Tracks the best validation loss and counts the epochs since it improved.
#[derive(Debug, Clone, PartialEq)]
pub struct EarlyStopper {
    patience: usize,
    min_delta: f64,
    best: Checkpoint,
    epochs_without_improvement: usize,
}

impl EarlyStopper {
    /// Starts with the network before training (epoch 0) as the best one.
    ///
    /// Panics when `metrics` has no validation set.
    pub fn new(settings: EarlyStopping, metrics: SplitMetrics, network: &Mlp) -> Self {
        assert!(
            metrics.validation.is_some(),
            "early stopping needs a validation set"
        );
        Self {
            patience: settings.patience,
            min_delta: settings.min_delta,
            best: Checkpoint {
                epoch: 0,
                metrics,
                network: network.clone(),
            },
            epochs_without_improvement: 0,
        }
    }

    /// Records the `metrics` of `network` after `epoch` and tells whether
    /// training should stop.
    ///
    /// The epoch improves when its validation loss is lower than the best one
    /// by more than `min_delta`; it then becomes the best epoch. Training
    /// stops after `patience` epochs in a row without improvement.
    pub fn should_stop(&mut self, epoch: usize, metrics: SplitMetrics, network: &Mlp) -> bool {
        let loss = validation_loss(&metrics);
        // A NaN best loss (diverged validation) is beaten by any number.
        let best = validation_loss(&self.best.metrics);
        if loss < best - self.min_delta || (best.is_nan() && !loss.is_nan()) {
            self.best = Checkpoint {
                epoch,
                metrics,
                network: network.clone(),
            };
            self.epochs_without_improvement = 0;
        } else {
            self.epochs_without_improvement += 1;
        }
        self.epochs_without_improvement >= self.patience
    }

    /// The best epoch seen so far.
    pub fn best(&self) -> &Checkpoint {
        &self.best
    }

    /// Takes the best epoch, e.g. to restore its network after training.
    pub fn into_best(self) -> Checkpoint {
        self.best
    }
}

fn validation_loss(metrics: &SplitMetrics) -> f64 {
    metrics
        .validation
        .expect("early stopping needs a validation set")
        .loss
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;
    use crate::network::{Activation, Layer};
    use crate::training::Metrics;

    fn metrics(validation_loss: f64) -> SplitMetrics {
        let metrics = |loss| Metrics {
            loss,
            accuracy: None,
            mse: None,
            mae: None,
        };
        SplitMetrics {
            train: metrics(1.0),
            validation: Some(metrics(validation_loss)),
            test: metrics(1.0),
        }
    }

    /// A one-weight network that tells epochs apart.
    fn network(weight: f64) -> Mlp {
        Mlp::from_layers(vec![Layer::from_parts(
            array![[weight]],
            None,
            Activation::Identity,
        )])
    }

    /// Feeds validation losses of epochs 1, 2, ... and returns the epoch at
    /// which training stops (if any) and the best epoch.
    fn run(patience: usize, min_delta: f64, losses: &[f64]) -> (Option<usize>, usize) {
        let settings = EarlyStopping {
            patience,
            min_delta,
        };
        let mut stopper = EarlyStopper::new(settings, metrics(1.0), &network(0.0));
        for (index, &loss) in losses.iter().enumerate() {
            let epoch = index + 1;
            if stopper.should_stop(epoch, metrics(loss), &network(epoch as f64)) {
                return (Some(epoch), stopper.best().epoch);
            }
        }
        (None, stopper.best().epoch)
    }

    #[test]
    fn stops_after_patience_epochs_without_improvement() {
        assert_eq!(run(3, 0.0, &[0.8, 0.9, 0.85, 0.95]), (Some(4), 1));
    }

    #[test]
    fn improvement_resets_patience() {
        assert_eq!(run(2, 0.0, &[0.8, 0.9, 0.7, 0.9, 0.8]), (Some(5), 3));
        assert_eq!(run(2, 0.0, &[0.8, 0.9, 0.7, 0.6]), (None, 4));
    }

    #[test]
    fn equal_loss_is_no_improvement() {
        assert_eq!(run(1, 0.0, &[1.0]), (Some(1), 0));
    }

    #[test]
    fn improvement_must_exceed_min_delta() {
        // Starting from 1.0, an improvement needs a loss below 0.9.
        assert_eq!(run(2, 0.1, &[0.95, 0.92]), (Some(2), 0));
        // From 0.85 on, it needs a loss below 0.75.
        assert_eq!(run(2, 0.1, &[0.85, 0.8, 0.78]), (Some(3), 1));
        assert_eq!(run(2, 0.1, &[0.85, 0.8, 0.7]), (None, 3));
    }

    #[test]
    fn any_loss_beats_nan() {
        let settings = EarlyStopping {
            patience: 1,
            min_delta: 0.0,
        };
        let mut stopper = EarlyStopper::new(settings, metrics(f64::NAN), &network(0.0));
        assert!(!stopper.should_stop(1, metrics(5.0), &network(1.0)));
        assert!(stopper.should_stop(2, metrics(f64::NAN), &network(2.0)));
        assert_eq!(stopper.best().epoch, 1);
    }

    #[test]
    fn keeps_network_and_metrics_of_best_epoch() {
        let settings = EarlyStopping {
            patience: 5,
            min_delta: 0.0,
        };
        let mut stopper = EarlyStopper::new(settings, metrics(1.0), &network(0.0));
        for (epoch, loss) in [(1, 0.9), (2, 0.5), (3, 0.7)] {
            stopper.should_stop(epoch, metrics(loss), &network(epoch as f64));
        }
        let best = stopper.into_best();
        assert_eq!(best.epoch, 2);
        assert_eq!(best.metrics, metrics(0.5));
        assert_eq!(best.network, network(2.0));
    }

    #[test]
    #[should_panic(expected = "early stopping needs a validation set")]
    fn needs_validation_metrics() {
        let mut without_validation = metrics(1.0);
        without_validation.validation = None;
        let settings = EarlyStopping {
            patience: 1,
            min_delta: 0.0,
        };
        EarlyStopper::new(settings, without_validation, &network(0.0));
    }
}
