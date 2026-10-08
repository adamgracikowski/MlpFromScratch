//! Seeded random number generators.
//!
//! Every source of randomness gets its own ChaCha stream derived from the
//! experiment seed. Streams are independent, so e.g. changing the training
//! subset does not change the initial weights.

use rand::SeedableRng;
use rand::rngs::ChaCha8Rng;

/// Purpose of a random stream. The discriminant is the ChaCha stream id and
/// must stay stable, otherwise old seeds stop reproducing old runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RngStream {
    /// Training subset and validation split.
    Data = 0,
}

/// Creates the generator of `stream` for the experiment `seed`.
pub fn rng(seed: u64, stream: RngStream) -> ChaCha8Rng {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(stream as u64);
    rng
}

#[cfg(test)]
mod tests {
    use rand::RngExt;

    use super::*;

    fn draw(seed: u64) -> Vec<u32> {
        let mut rng = rng(seed, RngStream::Data);
        (0..8).map(|_| rng.random()).collect()
    }

    #[test]
    fn same_seed_gives_same_numbers() {
        assert_eq!(draw(42), draw(42));
    }

    #[test]
    fn different_seeds_give_different_numbers() {
        assert_ne!(draw(42), draw(43));
    }

    #[test]
    fn stream_ids_are_stable() {
        assert_eq!(RngStream::Data as u64, 0);
        let mut other_stream = ChaCha8Rng::seed_from_u64(42);
        other_stream.set_stream(1);
        let other: Vec<u32> = (0..8).map(|_| other_stream.random()).collect();
        assert_ne!(draw(42), other);
    }
}
