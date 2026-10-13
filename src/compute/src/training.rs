//! Training of the network: mini-batch gradient descent with an optimizer,
//! metrics after every epoch and progress reported to an observer.

mod batches;
mod metrics;
mod observer;
mod optimizer;
mod outcome;

use ndarray::Axis;

pub use batches::BatchOrder;
pub use metrics::{Metrics, SplitMetrics};
pub use observer::{EpochRecord, IterationRecord, TrainingObserver};
pub use optimizer::OptimizerState;
pub use outcome::{StopReason, TrainingOutcome};

use crate::config::{Optimizer, TrainingConfig};
use crate::data::PreparedData;
use crate::network::Mlp;
use crate::rng::{RngStream, rng};

/// Trains `network` in place on `data.train` for `training.epochs` epochs.
///
/// Every epoch visits the training samples in a new random order (drawn
/// from `training.seed`) in mini-batches of `training.batch_size`; each
/// batch gives one `optimizer` step. All splits are evaluated before
/// training and after every epoch. Training stops early when the training
/// loss becomes NaN or infinite.
pub fn train(
    network: &mut Mlp,
    data: &PreparedData,
    optimizer: Optimizer,
    training: &TrainingConfig,
    observer: &mut dyn TrainingObserver,
) -> TrainingOutcome {
    let loss = training.loss;
    let samples = &data.train;
    let mut optimizer = OptimizerState::new(optimizer, network);
    let mut order = BatchOrder::new(samples.len());
    let mut rng = rng(training.seed, RngStream::Shuffle);
    let mut iteration = 0;
    let mut epochs = 0;

    let mut metrics = SplitMetrics::evaluate(network, data, loss);
    observer.on_epoch(
        &EpochRecord {
            epoch: 0,
            iteration,
            metrics,
        },
        network,
    );

    let stop = 'training: {
        if !metrics.train.loss.is_finite() {
            break 'training StopReason::Diverged {
                epoch: 0,
                iteration,
            };
        }
        for epoch in 1..=training.epochs {
            order.shuffle(&mut rng);
            for batch in order.batches(training.batch_size) {
                let inputs = samples.inputs.select(Axis(0), batch);
                let targets = samples.targets.select(Axis(0), batch);

                // Forward pass, keeping Z and A of every layer.
                let cache = network.forward(inputs.view());
                let output = cache
                    .layers
                    .last()
                    .expect("a network has at least one layer");
                let batch_loss = loss.value(&output.z, &output.a, &targets);
                if !batch_loss.is_finite() {
                    metrics = SplitMetrics::evaluate(network, data, loss);
                    break 'training StopReason::Diverged { epoch, iteration };
                }

                // Backward pass and one optimizer step.
                let gradients = network.backward(&cache, &targets, loss);
                optimizer.step(network, &gradients);
                iteration += 1;
                observer.on_iteration(
                    &IterationRecord {
                        epoch,
                        iteration,
                        batch_size: batch.len(),
                        loss: batch_loss,
                    },
                    network,
                );
            }

            epochs = epoch;
            metrics = SplitMetrics::evaluate(network, data, loss);
            observer.on_epoch(
                &EpochRecord {
                    epoch,
                    iteration,
                    metrics,
                },
                network,
            );
            if !metrics.train.loss.is_finite() {
                break 'training StopReason::Diverged { epoch, iteration };
            }
        }
        StopReason::Completed
    };

    TrainingOutcome {
        stop,
        epochs,
        iterations: iteration,
        metrics,
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, array};
    use rand::RngExt;

    use super::*;
    use crate::config::{CLASSIFICATION, ExperimentConfig, Loss, Normalization};
    use crate::data::{Dataset, Scaler};
    use crate::network::{Activation, Layer};

    /// Collects everything the training loop reports.
    #[derive(Default)]
    struct Recorder {
        iterations: Vec<IterationRecord>,
        epochs: Vec<EpochRecord>,
    }

    impl TrainingObserver for Recorder {
        fn on_iteration(&mut self, record: &IterationRecord, _network: &Mlp) {
            self.iterations.push(*record);
        }

        fn on_epoch(&mut self, record: &EpochRecord, _network: &Mlp) {
            self.epochs.push(*record);
        }
    }

    /// Points in `[-1, 1]^2`, class 1 above the line `x + y = 0`.
    fn separable(samples: usize, seed: u64) -> Dataset {
        let mut generator = rng(seed, RngStream::Data);
        let inputs = Array2::from_shape_fn((samples, 2), |_| generator.random_range(-1.0..1.0));
        let labels: Array1<usize> = inputs
            .rows()
            .into_iter()
            .map(|row| usize::from(row[0] + row[1] > 0.0))
            .collect();
        let targets = Array2::from_shape_fn((samples, 2), |(row, class)| {
            f64::from(u8::from(labels[row] == class))
        });
        Dataset {
            inputs,
            targets,
            labels: Some(labels),
        }
    }

    fn classification_data(validation: bool) -> PreparedData {
        let train = separable(200, 1);
        PreparedData {
            input_scaler: Scaler::fit(Normalization::None, &train.inputs),
            train,
            validation: validation.then(|| separable(50, 2)),
            test: separable(100, 3),
            feature_names: vec!["x".to_owned(), "y".to_owned()],
            target_scaler: None,
            classes: None,
        }
    }

    fn config() -> ExperimentConfig {
        toml::from_str(CLASSIFICATION).unwrap()
    }

    /// Trains the network of the fixture config (2 -> 16 -> 16 -> 2).
    fn run(
        config: &ExperimentConfig,
        data: &PreparedData,
        recorder: &mut Recorder,
    ) -> (Mlp, TrainingOutcome) {
        let mut init = rng(config.training.seed, RngStream::Init);
        let mut network = Mlp::new(&config.model, 2, 2, &mut init);
        let outcome = train(
            &mut network,
            data,
            config.optimizer,
            &config.training,
            recorder,
        );
        (network, outcome)
    }

    #[test]
    fn every_epoch_visits_every_batch() {
        let mut config = config();
        config.training.epochs = 3;
        config.training.batch_size = 16;
        let mut recorder = Recorder::default();
        let (_, outcome) = run(&config, &classification_data(false), &mut recorder);

        // 200 samples in batches of 16: 12 full batches and one of 8.
        assert_eq!(outcome.stop, StopReason::Completed);
        assert_eq!((outcome.epochs, outcome.iterations), (3, 3 * 13));
        assert_eq!(recorder.iterations.len(), 3 * 13);
        let sizes: Vec<usize> = recorder.iterations[..13]
            .iter()
            .map(|record| record.batch_size)
            .collect();
        assert_eq!(sizes, [[16; 12].as_slice(), &[8]].concat());
        let numbers: Vec<(usize, usize)> = recorder
            .iterations
            .iter()
            .map(|record| (record.epoch, record.iteration))
            .collect();
        assert_eq!(numbers[12..15], [(1, 13), (2, 14), (2, 15)]);

        let epochs: Vec<(usize, usize)> = recorder
            .epochs
            .iter()
            .map(|record| (record.epoch, record.iteration))
            .collect();
        assert_eq!(epochs, [(0, 0), (1, 13), (2, 26), (3, 39)]);
        assert_eq!(outcome.metrics, recorder.epochs[3].metrics);
    }

    #[test]
    fn same_seed_gives_same_training() {
        let mut config = config();
        config.training.epochs = 5;
        let data = classification_data(true);
        let mut first = Recorder::default();
        let mut second = Recorder::default();
        let (first_network, first_outcome) = run(&config, &data, &mut first);
        let (second_network, second_outcome) = run(&config, &data, &mut second);
        assert_eq!(first_network, second_network);
        assert_eq!(first_outcome, second_outcome);
        assert_eq!(first.iterations, second.iterations);
        assert_eq!(first.epochs, second.epochs);

        config.training.seed += 1;
        let (other_network, _) = run(&config, &data, &mut Recorder::default());
        assert_ne!(first_network, other_network);
    }

    #[test]
    fn same_network_with_other_seed_sees_other_batches() {
        let config = config();
        let data = classification_data(false);
        let mut network = Mlp::new(&config.model, 2, 2, &mut rng(1, RngStream::Init));
        let mut other = network.clone();
        let mut training = config.training.clone();
        training.epochs = 1;
        train(&mut network, &data, config.optimizer, &training, &mut ());
        training.seed += 1;
        train(&mut other, &data, config.optimizer, &training, &mut ());
        assert_ne!(network, other);
    }

    #[test]
    fn learns_separable_classes() {
        let mut config = config();
        config.training.epochs = 30;
        config.optimizer = Optimizer::Adam {
            learning_rate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        };
        let mut recorder = Recorder::default();
        let (_, outcome) = run(&config, &classification_data(true), &mut recorder);

        let initial = recorder.epochs[0].metrics;
        let last = outcome.metrics;
        assert!(last.train.loss < 0.2 * initial.train.loss);
        assert!(last.train.accuracy.unwrap() > 0.95, "{last:?}");
        assert!(last.test.accuracy.unwrap() > 0.9, "{last:?}");
        assert!(last.validation.unwrap().accuracy.unwrap() > 0.9, "{last:?}");
    }

    /// `y = 3x` fitted by one linear neuron; too large a learning rate makes
    /// every SGD step overshoot more, until the loss overflows.
    #[test]
    fn stops_when_loss_diverges() {
        let x = Array2::from_shape_fn((50, 1), |(row, _)| row as f64 / 25.0 - 1.0);
        let samples = Dataset {
            targets: x.mapv(|x| 3.0 * x),
            inputs: x,
            labels: None,
        };
        let data = PreparedData {
            input_scaler: Scaler::fit(Normalization::None, &samples.inputs),
            target_scaler: Some(Scaler::fit(Normalization::None, &samples.targets)),
            validation: None,
            test: samples.clone(),
            train: samples,
            feature_names: vec!["x".to_owned()],
            classes: None,
        };
        let mut config = config();
        config.training.loss = Loss::Mse;
        config.training.epochs = 1000;
        config.training.batch_size = 10;
        let mut network = Mlp::from_layers(vec![Layer::from_parts(
            array![[0.0]],
            Some(array![0.0]),
            Activation::Identity,
        )]);
        let mut recorder = Recorder::default();
        let outcome = train(
            &mut network,
            &data,
            Optimizer::Sgd { learning_rate: 1e3 },
            &config.training,
            &mut recorder,
        );

        let StopReason::Diverged { epoch, iteration } = outcome.stop else {
            panic!("expected divergence, got {outcome:?}");
        };
        assert!(epoch < 1000);
        assert_eq!(iteration, outcome.iterations);
        assert_eq!(recorder.iterations.len(), iteration);
        assert!(
            recorder
                .iterations
                .iter()
                .all(|record| record.loss.is_finite())
        );
        assert!(outcome.epochs <= epoch);
        assert!(!outcome.metrics.train.loss.is_finite());
        assert!(outcome.metrics.train.mse.is_some());
    }
}
