//! How a training run ended.

use super::SplitMetrics;

/// Why training stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// All configured epochs were run.
    Completed,
    /// The validation loss did not improve for `patience` epochs.
    EarlyStopped,
    /// The training loss (of a batch or of the whole training set) became
    /// NaN or infinite in `epoch` (0: before training), after `iteration`
    /// optimizer steps.
    Diverged { epoch: usize, iteration: usize },
}

/// Result of a training run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainingOutcome {
    pub stop: StopReason,
    /// Number of finished epochs.
    pub epochs: usize,
    /// Number of optimizer steps taken.
    pub iterations: usize,
    /// With early stopping, the epoch whose network is returned (whatever
    /// the stop reason); without it the network of the last step is returned.
    pub best_epoch: Option<usize>,
    /// Metrics of the returned network.
    pub metrics: SplitMetrics,
}
