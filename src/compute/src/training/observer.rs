//! Hooks through which the training loop reports its progress.

use super::SplitMetrics;
use crate::network::Mlp;

/// One optimizer step on one mini-batch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IterationRecord {
    /// Epoch the step belongs to, counted from 1.
    pub epoch: usize,
    /// Number of the step since the start of training, counted from 1.
    pub iteration: usize,
    /// Number of samples in the batch.
    pub batch_size: usize,
    /// Mean loss of the batch, computed before the step.
    pub loss: f64,
}

/// Metrics of the network after an epoch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpochRecord {
    /// Number of finished epochs; 0 is the network before training.
    pub epoch: usize,
    /// Number of steps taken so far.
    pub iteration: usize,
    pub metrics: SplitMetrics,
}

/// Receives the progress of training, e.g. to print it or to write
/// artifacts. Every method does nothing by default.
pub trait TrainingObserver {
    /// Called after every optimizer step with the updated `network`.
    fn on_iteration(&mut self, _record: &IterationRecord, _network: &Mlp) {}

    /// Called before training (epoch 0) and after every epoch.
    fn on_epoch(&mut self, _record: &EpochRecord, _network: &Mlp) {}

    /// Called once when training ends, after `iteration` optimizer steps,
    /// with the network of the last step: before early stopping restores
    /// the best one.
    fn on_stop(&mut self, _iteration: usize, _network: &Mlp) {}
}

/// Ignores all progress.
impl TrainingObserver for () {}

/// Passes all progress to the observer, if there is one.
impl<T: TrainingObserver> TrainingObserver for Option<T> {
    fn on_iteration(&mut self, record: &IterationRecord, network: &Mlp) {
        if let Some(observer) = self {
            observer.on_iteration(record, network);
        }
    }

    fn on_epoch(&mut self, record: &EpochRecord, network: &Mlp) {
        if let Some(observer) = self {
            observer.on_epoch(record, network);
        }
    }

    fn on_stop(&mut self, iteration: usize, network: &Mlp) {
        if let Some(observer) = self {
            observer.on_stop(iteration, network);
        }
    }
}

/// Passes all progress to both observers, first to `A`.
impl<A: TrainingObserver, B: TrainingObserver> TrainingObserver for (A, B) {
    fn on_iteration(&mut self, record: &IterationRecord, network: &Mlp) {
        self.0.on_iteration(record, network);
        self.1.on_iteration(record, network);
    }

    fn on_epoch(&mut self, record: &EpochRecord, network: &Mlp) {
        self.0.on_epoch(record, network);
        self.1.on_epoch(record, network);
    }

    fn on_stop(&mut self, iteration: usize, network: &Mlp) {
        self.0.on_stop(iteration, network);
        self.1.on_stop(iteration, network);
    }
}
