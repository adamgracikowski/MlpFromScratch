//! Order of the training samples and its split into mini-batches.

use rand::Rng;
use rand::seq::SliceRandom;

/// Permutation of the training sample indices, reshuffled every epoch and
/// cut into consecutive mini-batches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchOrder {
    indices: Vec<usize>,
}

impl BatchOrder {
    /// Visits `samples` samples in their original order until shuffled.
    pub fn new(samples: usize) -> Self {
        Self {
            indices: (0..samples).collect(),
        }
    }

    /// Draws a new random order from `rng` (Fisher-Yates shuffle).
    pub fn shuffle<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        self.indices.shuffle(rng);
    }

    /// Consecutive batches of `batch_size` sample indices; the last one is
    /// smaller when `batch_size` does not divide the number of samples.
    pub fn batches(&self, batch_size: usize) -> std::slice::Chunks<'_, usize> {
        self.indices.chunks(batch_size)
    }

    /// Number of batches per epoch: `ceil(samples / batch_size)`.
    pub fn batch_count(&self, batch_size: usize) -> usize {
        self.indices.len().div_ceil(batch_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::{RngStream, rng};

    fn shuffled(samples: usize, seed: u64) -> BatchOrder {
        let mut order = BatchOrder::new(samples);
        order.shuffle(&mut rng(seed, RngStream::Shuffle));
        order
    }

    #[test]
    fn batches_cover_every_sample_once() {
        let order = shuffled(10, 42);
        let mut visited: Vec<usize> = order.batches(4).flatten().copied().collect();
        visited.sort_unstable();
        assert_eq!(visited, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn last_batch_is_smaller() {
        let order = shuffled(10, 42);
        let sizes: Vec<usize> = order.batches(4).map(<[usize]>::len).collect();
        assert_eq!(sizes, [4, 4, 2]);
        assert_eq!(order.batch_count(4), 3);
        assert_eq!(order.batch_count(5), 2);
        assert_eq!(order.batch_count(32), 1);
    }

    #[test]
    fn same_seed_gives_same_order() {
        assert_eq!(shuffled(50, 42), shuffled(50, 42));
        assert_ne!(shuffled(50, 42), shuffled(50, 43));
    }

    #[test]
    fn every_epoch_has_a_new_order() {
        let mut rng = rng(42, RngStream::Shuffle);
        let mut order = BatchOrder::new(50);
        order.shuffle(&mut rng);
        let first = order.clone();
        order.shuffle(&mut rng);
        assert_ne!(order, first);
        assert_ne!(first, BatchOrder::new(50));
    }
}
