//! How a training run ended.

use super::SplitMetrics;

/// Why training stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// All configured epochs were run.
    Completed,
    /// The training loss (of a batch or of the whole training set) became
    /// NaN or infinite in `epoch` (0: before training), after `iteration`
    /// optimizer steps. The network is left as it was then.
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
    /// Metrics of the network returned by training.
    pub metrics: SplitMetrics,
}
