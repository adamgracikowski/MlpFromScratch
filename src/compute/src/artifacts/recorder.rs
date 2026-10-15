//! The training observer that collects the history written to the artifacts.

use super::{LossHistory, WeightHistory, WeightSample};
use crate::config::LoggingConfig;
use crate::network::Mlp;
use crate::rng::{RngStream, rng};
use crate::training::{EpochRecord, IterationRecord, TrainingObserver};

/// Collects the loss history, the weight snapshots and the metrics of every
/// epoch in memory. They are written to disk after training.
#[derive(Debug, Clone, PartialEq)]
pub struct Recorder {
    loss: LossHistory,
    weights: WeightHistory,
    epochs: Vec<EpochRecord>,
}

impl Recorder {
    /// A recorder for `network` with the intervals of `logging`. The
    /// recorded weights are drawn from the [`RngStream::WeightLog`] stream
    /// of `seed`.
    pub fn new(network: &Mlp, logging: &LoggingConfig, seed: u64) -> Self {
        let sample = WeightSample::draw(
            network,
            logging.max_weights_per_layer,
            &mut rng(seed, RngStream::WeightLog),
        );
        Self {
            loss: LossHistory::new(logging.loss_every),
            weights: WeightHistory::new(sample, logging.weights_every),
            epochs: Vec::new(),
        }
    }

    /// Loss averaged over windows of `loss_every` iterations.
    pub fn loss(&self) -> &LossHistory {
        &self.loss
    }

    /// Snapshots of the sampled weights and of the weight norms.
    pub fn weights(&self) -> &WeightHistory {
        &self.weights
    }

    /// Metrics before training (epoch 0) and after every epoch.
    pub fn epochs(&self) -> &[EpochRecord] {
        &self.epochs
    }
}

impl TrainingObserver for Recorder {
    fn on_iteration(&mut self, record: &IterationRecord, network: &Mlp) {
        self.loss.record(record);
        self.weights.on_iteration(record.iteration, network);
    }

    fn on_epoch(&mut self, record: &EpochRecord, network: &Mlp) {
        if record.epoch == 0 {
            self.weights.snapshot(0, network);
        }
        self.epochs.push(*record);
    }

    fn on_stop(&mut self, iteration: usize, network: &Mlp) {
        self.loss.finish();
        self.weights.snapshot(iteration, network);
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::*;
    use crate::config::{CLASSIFICATION, ExperimentConfig, Normalization};
    use crate::data::{Dataset, PreparedData, Scaler};
    use crate::training::train;

    fn config() -> ExperimentConfig {
        toml::from_str(CLASSIFICATION).unwrap()
    }

    /// 50 points on a line, class 1 for positive `x`.
    fn data() -> PreparedData {
        let dataset = Dataset::new(
            Array2::from_shape_fn((50, 2), |(row, column)| {
                (row as f64 - 24.5) * (column + 1) as f64
            }),
            Array2::from_shape_fn((50, 2), |(row, class)| {
                f64::from(u8::from((row >= 25) == (class == 1)))
            }),
            Some((0..50).map(|row| usize::from(row >= 25)).collect()),
        );
        PreparedData {
            input_scaler: Scaler::fit(Normalization::None, &dataset.inputs),
            train: dataset.clone(),
            validation: None,
            test: dataset,
            feature_names: vec!["x".to_owned(), "y".to_owned()],
            target_scaler: None,
            classes: None,
        }
    }

    #[test]
    fn records_whole_training_run() {
        let mut config = config();
        config.training.epochs = 3;
        config.training.batch_size = 16;
        config.logging.loss_every = 3;
        config.logging.weights_every = 5;
        config.logging.max_weights_per_layer = 10;
        let mut network = Mlp::new(&config.model, 2, 2, &mut rng(1, RngStream::Init));
        let mut recorder = Recorder::new(&network, &config.logging, config.training.seed);
        let outcome = train(
            &mut network,
            &data(),
            config.optimizer,
            &config.training,
            &mut recorder,
        );

        // 50 samples in batches of 16: 4 iterations per epoch, 12 in total.
        assert_eq!(outcome.iterations, 12);
        assert_eq!(recorder.loss().iterations(), [3, 6, 9, 12]);
        assert_eq!(recorder.loss().epochs(), [1, 2, 3, 3]);
        assert_eq!(recorder.weights().iterations(), [0, 5, 10, 12]);
        assert_eq!(recorder.epochs().len(), 4);

        // 2 -> 16 -> 16 -> 2: every layer has more than 10 parameters.
        let layers = recorder.weights().sample().layers().len();
        assert_eq!(layers, 3);
        for layer in 0..layers {
            assert_eq!(recorder.weights().values(layer).len(), 4 * 10);
        }
        assert_eq!(recorder.weights().weight_norms().len(), 4 * layers);
        let last = &recorder.weights().values(2)[30..];
        let mut expected = Vec::new();
        recorder
            .weights()
            .sample()
            .read(2, &network.layers()[2], &mut expected);
        assert_eq!(last, expected);
    }
}
