//! Human-readable summary of a loaded experiment config.

use std::fmt;

use crate::config::{LoadedConfig, Task};

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
        )
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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::config::{CLASSIFICATION, ExperimentConfig};

    fn loaded(config: ExperimentConfig) -> LoadedConfig {
        LoadedConfig {
            path: PathBuf::from("experiment/config.toml"),
            config,
            warnings: Vec::new(),
            feature_count: 2,
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
    }
}
