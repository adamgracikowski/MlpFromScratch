//! Snapshots of the recorded weights and of the weight norms.

use super::WeightSample;
use crate::network::Mlp;

/// Values of the sampled parameters and the norms of every layer, taken
/// before training, every `every` iterations and after the last iteration.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightHistory {
    sample: WeightSample,
    every: usize,
    /// Iteration of every snapshot; 0 is the network before training.
    iterations: Vec<usize>,
    /// Sampled values of every layer, one row of the layer's sample size
    /// per snapshot (row-major).
    values: Vec<Vec<f64>>,
    /// Frobenius norm of `W` of every layer, one row per snapshot.
    weight_norms: Vec<f64>,
    /// Euclidean norm of `b` of every layer (NaN without bias), one row per
    /// snapshot.
    bias_norms: Vec<f64>,
}

impl WeightHistory {
    /// An empty history of the parameters in `sample`, with snapshots every
    /// `every` iterations (`every >= 1`).
    pub fn new(sample: WeightSample, every: usize) -> Self {
        assert!(every > 0, "the snapshot interval must be at least 1");
        let layers = sample.layers().len();
        Self {
            sample,
            every,
            iterations: Vec::new(),
            values: vec![Vec::new(); layers],
            weight_norms: Vec::new(),
            bias_norms: Vec::new(),
        }
    }

    /// Takes a snapshot when `iteration` is a multiple of the interval.
    pub fn on_iteration(&mut self, iteration: usize, network: &Mlp) {
        if iteration % self.every == 0 {
            self.snapshot(iteration, network);
        }
    }

    /// Takes a snapshot of `network` after `iteration` steps, unless the last
    /// snapshot was taken at the same iteration.
    pub fn snapshot(&mut self, iteration: usize, network: &Mlp) {
        if self.iterations.last() == Some(&iteration) {
            return;
        }
        self.iterations.push(iteration);
        for (index, layer) in network.layers().iter().enumerate() {
            self.sample.read(index, layer, &mut self.values[index]);
            self.weight_norms.push(norm(layer.weights().iter()));
            self.bias_norms
                .push(layer.bias().map_or(f64::NAN, |bias| norm(bias.iter())));
        }
    }

    /// The recorded parameters.
    pub fn sample(&self) -> &WeightSample {
        &self.sample
    }

    /// Iteration of every snapshot.
    pub fn iterations(&self) -> &[usize] {
        &self.iterations
    }

    /// Sampled values of layer `index`: snapshots x sample size, row-major.
    pub fn values(&self, index: usize) -> &[f64] {
        &self.values[index]
    }

    /// Frobenius norms of `W`: snapshots x layers, row-major.
    pub fn weight_norms(&self) -> &[f64] {
        &self.weight_norms
    }

    /// Euclidean norms of `b` (NaN without bias): snapshots x layers, row-major.
    pub fn bias_norms(&self) -> &[f64] {
        &self.bias_norms
    }
}

/// Euclidean norm of all `values` (the Frobenius norm of a matrix).
fn norm<'a>(values: impl Iterator<Item = &'a f64>) -> f64 {
    values.map(|value| value * value).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;
    use crate::network::{Activation, Layer};
    use crate::rng::{RngStream, rng};

    /// 1 -> 2 (bias) -> 1 (no bias), every parameter scaled by `factor`.
    fn network(factor: f64) -> Mlp {
        Mlp::from_layers(vec![
            Layer::from_parts(
                array![[3.0, 4.0]] * factor,
                Some(array![0.0, 2.0] * factor),
                Activation::Relu,
            ),
            Layer::from_parts(array![[1.0], [0.0]] * factor, None, Activation::Identity),
        ])
    }

    fn history(every: usize) -> WeightHistory {
        let sample = WeightSample::draw(&network(1.0), 10, &mut rng(1, RngStream::WeightLog));
        WeightHistory::new(sample, every)
    }

    #[test]
    fn snapshots_at_start_every_interval_and_end() {
        let mut history = history(2);
        history.snapshot(0, &network(1.0));
        for iteration in 1..=5 {
            history.on_iteration(iteration, &network(iteration as f64));
        }
        history.snapshot(5, &network(5.0));
        assert_eq!(history.iterations(), [0, 2, 4, 5]);
        // Layer 0: [W; b] = [[3, 4], [0, 2]] times the factor.
        assert_eq!(
            history.values(0),
            [
                [3.0, 4.0, 0.0, 2.0],
                [6.0, 8.0, 0.0, 4.0],
                [12.0, 16.0, 0.0, 8.0],
                [15.0, 20.0, 0.0, 10.0]
            ]
            .concat()
        );
        assert_eq!(history.values(1), [1.0, 0.0, 2.0, 0.0, 4.0, 0.0, 5.0, 0.0]);
    }

    #[test]
    fn last_snapshot_is_not_repeated() {
        let mut history = history(2);
        history.on_iteration(2, &network(1.0));
        history.snapshot(2, &network(1.0));
        assert_eq!(history.iterations(), [2]);
    }

    #[test]
    fn records_norms_of_every_layer() {
        let mut history = history(1);
        history.snapshot(0, &network(1.0));
        history.snapshot(1, &network(2.0));
        // ||[3, 4]|| = 5, ||[0, 2]|| = 2, ||[1, 0]|| = 1, no bias in layer 1.
        assert_eq!(history.weight_norms(), [5.0, 1.0, 10.0, 2.0]);
        let bias = history.bias_norms();
        assert_eq!((bias[0], bias[2]), (2.0, 4.0));
        assert!(bias[1].is_nan() && bias[3].is_nan());
    }
}
