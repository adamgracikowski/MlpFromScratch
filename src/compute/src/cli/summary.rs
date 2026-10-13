//! Human-readable summary of a loaded experiment config.

use std::fmt;
use std::time::Duration;

use super::progress::SplitLine;
use crate::config::{LoadedConfig, Task};
use crate::data::{Dataset, PreparedData};
use crate::network::Mlp;
use crate::training::{StopReason, TrainingOutcome};

/// Prints the warnings of a loaded config to stderr and its summary to stdout.
pub(super) fn report(loaded: &LoadedConfig) {
    for warning in &loaded.warnings {
        eprintln!("{:?}", miette::Report::new(warning.clone()));
    }
    println!("{}", Summary(loaded));
}

/// Summary of the experiment: task, data, architecture and training setup.
pub(super) struct Summary<'a>(pub &'a LoadedConfig);

impl fmt::Display for Summary<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let LoadedConfig {
            path,
            config,
            feature_count,
            ..
        } = self.0;
        let data = &config.data;
        let model = &config.model;
        let training = &config.training;

        writeln!(f, "config:     {}", path.display())?;
        writeln!(f, "task:       {}", config.task)?;
        writeln!(f, "train:      {}", data.train.display())?;
        writeln!(f, "test:       {}", data.test.display())?;
        writeln!(
            f,
            "columns:    {feature_count} feature(s), target `{}`",
            data.target
        )?;
        writeln!(
            f,
            "normalize:  inputs {}, target {}",
            data.input_normalization, data.target_normalization
        )?;
        if let Some(subset) = data.train_subset {
            writeln!(f, "subset:     {subset} training sample(s)")?;
        }
        if data.validation_fraction > 0.0 {
            writeln!(
                f,
                "validation: {} of training set",
                data.validation_fraction
            )?;
        }
        writeln!(f, "network:    {}", Architecture(self.0))?;
        writeln!(
            f,
            "layers:     hidden {}, output {}, init {}, bias {}",
            model.hidden_activation,
            model.output_activation,
            model.init,
            if model.bias { "on" } else { "off" }
        )?;
        writeln!(f, "loss:       {}", training.loss)?;
        writeln!(f, "optimizer:  {}", config.optimizer)?;
        write!(
            f,
            "training:   {} epoch(s), batch size {}, seed {}",
            training.epochs, training.batch_size, training.seed
        )?;
        if let Some(early_stopping) = &training.early_stopping {
            write!(
                f,
                "\nearly stop: patience {}, min_delta {}",
                early_stopping.patience, early_stopping.min_delta
            )?;
        }
        Ok(())
    }
}

/// Layer sizes from input to output, e.g. `2 -> 16 -> 16 -> classes`.
///
/// The number of classes is only known once the data is loaded.
struct Architecture<'a>(&'a LoadedConfig);

impl fmt::Display for Architecture<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.feature_count)?;
        for size in &self.0.config.model.hidden_layers {
            write!(f, " -> {size}")?;
        }
        match self.0.config.task {
            Task::Classification => write!(f, " -> classes"),
            Task::Regression => write!(f, " -> 1"),
        }
    }
}

/// Summary of the prepared data: sample counts, classes and normalization.
pub(super) struct DataSummary<'a>(pub &'a PreparedData);

impl fmt::Display for DataSummary<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let prepared = self.0;
        let splits: Vec<(&str, &Dataset)> = [
            ("train", Some(&prepared.train)),
            ("validation", prepared.validation.as_ref()),
            ("test", Some(&prepared.test)),
        ]
        .into_iter()
        .filter_map(|(name, dataset)| dataset.map(|dataset| (name, dataset)))
        .collect();

        let counts: Vec<String> = splits
            .iter()
            .map(|(name, dataset)| format!("{name} {}", dataset.len()))
            .collect();
        write!(f, "samples:    {}", counts.join(", "))?;
        if let Some(classes) = &prepared.classes {
            let labels: Vec<String> = classes.classes().iter().map(i64::to_string).collect();
            write!(f, "\nclasses:    {}", labels.join(", "))?;
            for (name, dataset) in &splits {
                let counts = dataset
                    .class_counts(classes.class_count())
                    .unwrap_or_default();
                let counts: Vec<String> = counts.iter().map(usize::to_string).collect();
                write!(f, "\n  {name:<11}{}", counts.join(" / "))?;
            }
        }
        if let Some(scaler) = &prepared.target_scaler {
            write!(
                f,
                "\ntarget:     {} (offset {:.4}, scale {:.4})",
                scaler.normalization, scaler.offset[0], scaler.scale[0]
            )?;
        }
        Ok(())
    }
}

/// Parameter counts of the built network, per layer as `in x out + bias`,
/// e.g. `parameters: 371 (2x16+16, 16x16+16, 16x3+3)`.
pub(super) struct NetworkSummary<'a>(pub &'a Mlp);

impl fmt::Display for NetworkSummary<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let layers: Vec<String> = self
            .0
            .layers()
            .iter()
            .map(|layer| {
                let weights = format!("{}x{}", layer.input_size(), layer.output_size());
                match layer.bias() {
                    Some(bias) => format!("{weights}+{}", bias.len()),
                    None => weights,
                }
            })
            .collect();
        write!(
            f,
            "parameters: {} ({})",
            self.0.parameter_count(),
            layers.join(", ")
        )
    }
}

/// How training ended, how long it took and the metrics of the returned
/// network, e.g. `stop:       completed after 200 epoch(s), ...`.
pub(super) struct TrainingSummary<'a>(pub &'a TrainingOutcome, pub Duration);

impl fmt::Display for TrainingSummary<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let TrainingSummary(outcome, elapsed) = *self;
        let steps = format!(
            "{} epoch(s), {} iteration(s)",
            outcome.epochs, outcome.iterations
        );
        match outcome.stop {
            StopReason::Completed => write!(f, "stop:       completed after {steps}")?,
            StopReason::EarlyStopped => write!(f, "stop:       early stopped after {steps}")?,
            StopReason::Diverged { epoch, iteration } => write!(
                f,
                "stop:       diverged in epoch {epoch} after {iteration} iteration(s)"
            )?,
        }
        writeln!(f, " in {:.2} s", elapsed.as_secs_f64())?;
        if let Some(best) = outcome.best_epoch {
            writeln!(f, "best epoch: {best} (weights restored)")?;
        }
        write!(f, "result:     {}", SplitLine(&outcome.metrics))
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use miette::NamedSource;
    use ndarray::{Array1, Array2};

    use super::*;
    use crate::config::{CLASSIFICATION, EarlyStopping, ExperimentConfig, Normalization};
    use crate::data::{ClassEncoding, Scaler, Table};
    use crate::network::{Activation, Layer};
    use crate::training::{Metrics, SplitMetrics};

    fn loaded(config: ExperimentConfig) -> LoadedConfig {
        LoadedConfig {
            path: PathBuf::from("experiment/config.toml"),
            config,
            warnings: Vec::new(),
            feature_count: 2,
            source: NamedSource::new("config.toml", String::new()),
        }
    }

    #[test]
    fn shows_classification_architecture() {
        let config: ExperimentConfig = toml::from_str(CLASSIFICATION).unwrap();
        let loaded = loaded(config);
        assert_eq!(
            Architecture(&loaded).to_string(),
            "2 -> 16 -> 16 -> classes"
        );
    }

    #[test]
    fn shows_regression_architecture_without_hidden_layers() {
        let mut config: ExperimentConfig = toml::from_str(CLASSIFICATION).unwrap();
        config.task = Task::Regression;
        config.model.hidden_layers.clear();
        assert_eq!(Architecture(&loaded(config)).to_string(), "2 -> 1");
    }

    #[test]
    fn summary_lists_key_settings() {
        let config: ExperimentConfig = toml::from_str(CLASSIFICATION).unwrap();
        let summary = Summary(&loaded(config)).to_string();
        for expected in [
            "task:       classification",
            "2 feature(s), target `cls`",
            "loss:       cross_entropy",
            "seed 42",
        ] {
            assert!(
                summary.contains(expected),
                "missing {expected:?} in\n{summary}"
            );
        }
        assert!(!summary.contains("subset:"));
        assert!(!summary.contains("early stop:"));
    }

    #[test]
    fn summary_lists_early_stopping() {
        let mut config: ExperimentConfig = toml::from_str(CLASSIFICATION).unwrap();
        config.training.early_stopping = Some(EarlyStopping {
            patience: 10,
            min_delta: 1e-4,
        });
        let summary = Summary(&loaded(config)).to_string();
        assert!(summary.ends_with("\nearly stop: patience 10, min_delta 0.0001"));
    }

    fn outcome(stop: StopReason, best_epoch: Option<usize>) -> TrainingOutcome {
        let metrics = Metrics {
            loss: 0.5,
            accuracy: Some(0.9),
            mse: None,
            mae: None,
        };
        TrainingOutcome {
            stop,
            epochs: 57,
            iterations: 912,
            best_epoch,
            metrics: SplitMetrics {
                train: metrics,
                validation: None,
                test: metrics,
            },
        }
    }

    #[test]
    fn training_summary_shows_stop_and_result() {
        let outcome = outcome(StopReason::EarlyStopped, Some(47));
        assert_eq!(
            TrainingSummary(&outcome, Duration::from_millis(1234)).to_string(),
            "stop:       early stopped after 57 epoch(s), 912 iteration(s) in 1.23 s\n\
             best epoch: 47 (weights restored)\n\
             result:     train loss 0.5000 acc 90.00% | test loss 0.5000 acc 90.00%"
        );
    }

    #[test]
    fn training_summary_shows_divergence() {
        let stop = StopReason::Diverged {
            epoch: 3,
            iteration: 41,
        };
        let summary = TrainingSummary(&outcome(stop, None), Duration::ZERO).to_string();
        assert!(summary.starts_with(
            "stop:       diverged in epoch 3 after 41 iteration(s) in 0.00 s\nresult:"
        ));
    }

    fn dataset(labels: Vec<usize>) -> Dataset {
        let rows = labels.len();
        Dataset {
            inputs: Array2::zeros((rows, 2)),
            targets: Array2::zeros((rows, 2)),
            labels: Some(Array1::from(labels)),
        }
    }

    #[test]
    fn data_summary_lists_samples_and_classes() {
        let table = Table {
            path: PathBuf::from("train.csv"),
            feature_names: vec!["x".to_owned()],
            features: Array2::zeros((2, 1)),
            target: vec![1.0, 2.0],
            lines: vec![2, 3],
        };
        let prepared = PreparedData {
            train: dataset(vec![0, 0, 1]),
            validation: Some(dataset(vec![1])),
            test: dataset(vec![0, 1]),
            feature_names: vec!["x".to_owned(), "y".to_owned()],
            input_scaler: Scaler::fit(Normalization::None, &Array2::zeros((1, 2))),
            target_scaler: None,
            classes: Some(ClassEncoding::fit(&table).unwrap()),
        };
        assert_eq!(
            DataSummary(&prepared).to_string(),
            "samples:    train 3, validation 1, test 2\n\
             classes:    1, 2\n  \
             train      2 / 1\n  \
             validation 0 / 1\n  \
             test       1 / 1"
        );
    }

    #[test]
    fn network_summary_lists_parameters_per_layer() {
        let network = Mlp::from_layers(vec![
            Layer::from_parts(
                Array2::zeros((2, 4)),
                Some(Array1::zeros(4)),
                Activation::Tanh,
            ),
            Layer::from_parts(Array2::zeros((4, 3)), None, Activation::Softmax),
        ]);
        assert_eq!(
            NetworkSummary(&network).to_string(),
            "parameters: 24 (2x4+4, 4x3)"
        );
    }
}
