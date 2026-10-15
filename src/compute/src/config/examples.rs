//! Checks that the example experiment configs in `data/experiments/` stay valid
//! and that their datasets can be prepared, fed through the network, both
//! forward and backward, and trained on.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use miette::NamedSource;

use super::{ExperimentConfig, Issue, LoadedConfig, check_files, validate};
use crate::artifacts::read_npy;
use crate::artifacts::{Recorder, Run, write_artifacts};
use crate::data::prepare;
use crate::network::Mlp;
use crate::rng::{RngStream, rng};
use crate::training::{SplitMetrics, StopReason, train};

/// Example configs, relative to the repository root.
const EXAMPLES: [&str; 2] = [
    "data/experiments/classification/example-three-gauss/config.toml",
    "data/experiments/regression/example-cube/config.toml",
];

/// The repository root; dataset paths in configs are relative to it.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn example_configs_are_valid() {
    let root = repo_root();
    for example in EXAMPLES {
        let path = root.join(example);
        let text = std::fs::read_to_string(&path).unwrap();
        let mut config: ExperimentConfig = toml::from_str(&text).unwrap();
        // Tests run in the crate directory, not in the repository root.
        config.data.train = root.join(&config.data.train);
        config.data.test = root.join(&config.data.test);

        let mut issues = validate(&config);
        let files = check_files(&config.data);
        issues.extend(files.issues);
        let issues: Vec<String> = issues.iter().map(Issue::to_string).collect();
        assert!(issues.is_empty(), "{example}: {issues:?}");

        let loaded = LoadedConfig {
            source: NamedSource::new(example, text),
            path,
            config,
            warnings: Vec::new(),
            feature_count: files.feature_count.unwrap(),
        };
        let prepared = match prepare(&loaded) {
            Ok(prepared) => prepared,
            Err(err) => panic!("{example}: {err:?}"),
        };

        let test = &prepared.test;
        let network = Mlp::new(
            &loaded.config.model,
            test.inputs.ncols(),
            test.targets.ncols(),
            &mut rng(loaded.config.training.seed, RngStream::Init),
        );
        let predictions = network.predict(test.inputs.view());
        assert_eq!(predictions.dim(), test.targets.dim(), "{example}");
        assert!(
            predictions.iter().all(|value| value.is_finite()),
            "{example}"
        );

        let loss = loaded.config.training.loss;
        let cache = network.forward(test.inputs.view());
        let output = cache.layers.last().unwrap();
        let value = loss.value(&output.z, &output.a, &test.targets);
        assert!(value.is_finite(), "{example}: loss {value}");
        let gradients = network.backward(&cache, &test.targets, loss);
        for (layer, gradient) in network.layers().iter().zip(&gradients.layers) {
            assert_eq!(gradient.weights.dim(), layer.weights().dim(), "{example}");
            assert!(gradient.weights.iter().all(|value| value.is_finite()));
        }

        // A few epochs of training lower the training loss.
        let mut training = loaded.config.training.clone();
        training.epochs = 3;
        let mut trained = network.clone();
        let optimizer = loaded.config.optimizer;
        let config = &loaded.config;
        let mut recorder = Recorder::new(&trained, &config.logging, training.seed);
        let outcome = train(&mut trained, &prepared, optimizer, &training, &mut recorder);
        assert_eq!(outcome.stop, StopReason::Completed, "{example}");
        let initial = SplitMetrics::evaluate(&network, &prepared, loss).train.loss;
        let last = outcome.metrics.train.loss;
        assert!(last < initial, "{example}: {initial} -> {last}");

        // The artifacts of the run hold a prediction grid over the inputs.
        let experiment = tempfile::TempDir::new().unwrap();
        let run = Run {
            config,
            data: &prepared,
            network: &trained,
            outcome: &outcome,
            recorder: &recorder,
            started_at: SystemTime::now(),
            training_time: Duration::ZERO,
        };
        let dir = write_artifacts(experiment.path(), &run).unwrap();
        let (shape, outputs) = read_npy::<f64>(&dir.join("grid/outputs.npy"));
        let resolution = config.output.grid_resolution as u64;
        let classes = test.targets.ncols() as u64;
        let expected = match test.inputs.ncols() {
            1 => vec![resolution, classes],
            _ => vec![resolution, resolution, classes],
        };
        assert_eq!(shape, expected, "{example}");
        assert!(outputs.iter().all(|value| value.is_finite()), "{example}");
        for file in [
            "epochs.csv",
            "metadata.json",
            "history/loss.npy",
            "weights/iteration.npy",
        ] {
            assert!(dir.join(file).is_file(), "{example}: missing {file}");
        }
    }
}
