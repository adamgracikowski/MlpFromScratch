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
}

/// Ignores all progress.
impl TrainingObserver for () {}
