//! Training loss per iteration, averaged over windows of `loss_every` steps.

use crate::training::IterationRecord;

/// Mean batch loss over windows of `every` iterations.
///
/// Each batch contributes its loss weighted by its size, so a window gives
/// the mean loss per sample over all of its batches; with `every = 1` it is
/// just the loss of every batch. A window ends at an iteration divisible by
/// `every`; the last, possibly shorter window ends with [`LossHistory::finish`].
#[derive(Debug, Clone, PartialEq)]
pub struct LossHistory {
    every: usize,
    /// Sum of `loss * batch_size` over the open window.
    window_loss: f64,
    /// Number of samples in the open window.
    window_samples: usize,
    /// The last step of the open window, if it has any.
    window_end: Option<IterationRecord>,
    iterations: Vec<usize>,
    epochs: Vec<usize>,
    losses: Vec<f64>,
}

impl LossHistory {
    /// An empty history with windows of `every` iterations (`every >= 1`).
    pub fn new(every: usize) -> Self {
        assert!(
            every > 0,
            "the loss window must have at least one iteration"
        );
        Self {
            every,
            window_loss: 0.0,
            window_samples: 0,
            window_end: None,
            iterations: Vec::new(),
            epochs: Vec::new(),
            losses: Vec::new(),
        }
    }

    /// Adds the batch loss of one optimizer step.
    pub fn record(&mut self, record: &IterationRecord) {
        self.window_loss += record.loss * record.batch_size as f64;
        self.window_samples += record.batch_size;
        self.window_end = Some(*record);
        if record.iteration % self.every == 0 {
            self.close_window();
        }
    }

    /// Closes the last window when training ends in the middle of one.
    pub fn finish(&mut self) {
        self.close_window();
    }

    fn close_window(&mut self) {
        let Some(end) = self.window_end.take() else {
            return;
        };
        self.iterations.push(end.iteration);
        self.epochs.push(end.epoch);
        self.losses
            .push(self.window_loss / self.window_samples as f64);
        self.window_loss = 0.0;
        self.window_samples = 0;
    }

    /// The last iteration of every window.
    pub fn iterations(&self) -> &[usize] {
        &self.iterations
    }

    /// The epoch of the last iteration of every window.
    pub fn epochs(&self) -> &[usize] {
        &self.epochs
    }

    /// The mean loss of every window.
    pub fn losses(&self) -> &[f64] {
        &self.losses
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Steps 1..=7 of two epochs (4 + 3 batches), the last batch of every
    /// epoch half the size; the loss of step `i` is `i`.
    fn steps() -> Vec<IterationRecord> {
        (1..=7)
            .map(|iteration| IterationRecord {
                epoch: if iteration <= 4 { 1 } else { 2 },
                iteration,
                batch_size: if iteration == 4 || iteration == 7 {
                    5
                } else {
                    10
                },
                loss: iteration as f64,
            })
            .collect()
    }

    fn history(every: usize) -> LossHistory {
        let mut history = LossHistory::new(every);
        for step in steps() {
            history.record(&step);
        }
        history.finish();
        history
    }

    #[test]
    fn window_of_one_keeps_every_batch_loss() {
        let history = history(1);
        assert_eq!(history.iterations(), [1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(history.epochs(), [1, 1, 1, 1, 2, 2, 2]);
        assert_eq!(history.losses(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn windows_average_weighted_by_batch_size() {
        let history = history(3);
        // Windows 1..=3, 4..=6 (crossing the epoch) and the partial 7..=7.
        assert_eq!(history.iterations(), [3, 6, 7]);
        assert_eq!(history.epochs(), [1, 2, 2]);
        let second = (4.0 * 5.0 + 5.0 * 10.0 + 6.0 * 10.0) / 25.0;
        assert_eq!(history.losses(), [2.0, second, 7.0]);
    }

    #[test]
    fn finish_adds_nothing_after_a_full_window() {
        let mut history = LossHistory::new(7);
        for step in steps() {
            history.record(&step);
        }
        history.finish();
        history.finish();
        assert_eq!(history.iterations(), [7]);
    }

    #[test]
    fn empty_training_gives_empty_history() {
        let mut history = LossHistory::new(2);
        history.finish();
        assert!(history.losses().is_empty());
    }
}
