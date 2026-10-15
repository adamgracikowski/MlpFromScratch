//! Training artifacts: everything the presentation layer needs to plot a
//! run, recorded during training and written to `<experiment>/artifacts/`.
//!
//! Arrays are `.npy` files, the metrics of every epoch a CSV table and the
//! description of the run `metadata.json`. The layout is documented in
//! `src/compute/README.md`.

mod epochs_csv;
mod error;
mod grid;
mod loss_history;
mod metadata;
mod npy;
mod predictions;
mod recorder;
mod weight_history;
mod weight_sample;

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

pub use error::ArtifactError;
pub use grid::Grid;
pub use loss_history::LossHistory;
pub use predictions::{Predictions, SplitPredictions, Targets, predict_in_chunks};
pub use recorder::Recorder;
pub use weight_history::WeightHistory;
pub use weight_sample::WeightSample;

use crate::config::ExperimentConfig;
use crate::data::PreparedData;
use crate::network::Mlp;
use crate::training::TrainingOutcome;
use metadata::Metadata;
#[cfg(test)]
pub(crate) use npy::read_npy;
use npy::write_npy;

/// Name of the artifacts directory inside an experiment directory.
pub const ARTIFACTS_DIR: &str = "artifacts";

/// A finished training run and everything recorded during it.
#[derive(Debug, Clone, Copy)]
pub struct Run<'a> {
    pub config: &'a ExperimentConfig,
    pub data: &'a PreparedData,
    /// The returned network: after early stopping, the best one.
    pub network: &'a Mlp,
    pub outcome: &'a TrainingOutcome,
    pub recorder: &'a Recorder,
    /// When the training loop started.
    pub started_at: SystemTime,
    /// Wall time of the training loop.
    pub training_time: Duration,
}

/// Writes the artifacts of `run` to `<experiment_dir>/artifacts/` and
/// returns that directory. Artifacts of a previous run are removed first,
/// so the directory never mixes files of two runs.
///
/// Predictions and the grid are computed on `run.network`.
pub fn write_artifacts(experiment_dir: &Path, run: &Run<'_>) -> Result<PathBuf, ArtifactError> {
    let dir = experiment_dir.join(ARTIFACTS_DIR);
    if dir.exists() {
        fs::remove_dir_all(&dir).map_err(|source| ArtifactError::Clear {
            path: dir.clone(),
            source,
        })?;
    }
    create_dir(&dir)?;

    epochs_csv::write_epochs_csv(&dir.join("epochs.csv"), run.recorder.epochs())?;
    let history = dir.join("history");
    create_dir(&history)?;
    run.recorder.loss().write(&history)?;
    let weights = dir.join("weights");
    create_dir(&weights)?;
    run.recorder.weights().write(&weights)?;
    write_model(&dir.join("model"), run.network)?;

    SplitPredictions::compute(run.network, run.data).write(&dir.join("predictions"))?;
    let grid = Grid::compute(run.network, run.data, &run.config.output);
    if let Some(grid) = &grid {
        grid.write(&dir.join("grid"))?;
    }
    Metadata::new(run, grid.as_ref()).write(&dir.join("metadata.json"))?;
    Ok(dir)
}

/// Writes `layer_<l>_weights.npy` (`in x out`) and, with bias,
/// `layer_<l>_bias.npy` (`out`) of every layer to `dir`.
fn write_model(dir: &Path, network: &Mlp) -> Result<(), ArtifactError> {
    create_dir(dir)?;
    for (index, layer) in network.layers().iter().enumerate() {
        let weights = layer.weights();
        let path = dir.join(format!("layer_{index}_weights.npy"));
        write_npy(&path, weights.shape(), weights.iter().copied())?;
        if let Some(bias) = layer.bias() {
            let path = dir.join(format!("layer_{index}_bias.npy"));
            write_npy(&path, bias.shape(), bias.iter().copied())?;
        }
    }
    Ok(())
}

/// Creates `dir` (and missing parents).
fn create_dir(dir: &Path) -> Result<(), ArtifactError> {
    fs::create_dir_all(dir).map_err(|source| ArtifactError::write(dir, source))
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use tempfile::TempDir;

    use super::npy::read_npy;
    use super::*;
    use crate::config::{CLASSIFICATION, Loss, Normalization, OutputActivation, Task};
    use crate::data::{Dataset, Scaler};
    use crate::rng::{RngStream, rng};
    use crate::training::train;

    /// Regression on 3 features: `y = x0 + x1 + x2` on 40 + 10 + 20 samples.
    fn data() -> PreparedData {
        let dataset = |samples: usize, shift: f64| {
            let inputs =
                Array2::from_shape_fn((samples, 3), |(row, column)| (row + column) as f64 + shift);
            let targets = inputs
                .sum_axis(ndarray::Axis(1))
                .insert_axis(ndarray::Axis(1));
            Dataset::new(inputs, targets, None)
        };
        let train = dataset(40, 0.0);
        PreparedData {
            input_scaler: Scaler::fit(Normalization::None, &train.inputs),
            target_scaler: Some(Scaler::fit(Normalization::None, &train.targets)),
            validation: Some(dataset(10, 0.5)),
            test: dataset(20, 0.25),
            train,
            feature_names: vec!["a".to_owned(), "b".to_owned(), "c".to_owned()],
            classes: None,
        }
    }

    #[test]
    fn writes_fresh_artifacts_without_grid_for_three_features() {
        let mut config: ExperimentConfig = toml::from_str(CLASSIFICATION).unwrap();
        config.task = Task::Regression;
        config.model.output_activation = OutputActivation::Identity;
        config.training.loss = Loss::Mse;
        config.training.epochs = 2;
        config.training.batch_size = 16;
        let data = data();
        let mut network = Mlp::new(&config.model, 3, 1, &mut rng(1, RngStream::Init));
        let mut recorder = Recorder::new(&network, &config.logging, config.training.seed);
        let outcome = train(
            &mut network,
            &data,
            config.optimizer,
            &config.training,
            &mut recorder,
        );

        let experiment = TempDir::new().unwrap();
        let stale = experiment.path().join(ARTIFACTS_DIR).join("grid");
        fs::create_dir_all(&stale).unwrap();
        fs::write(stale.join("x.npy"), "old run").unwrap();

        let run = Run {
            config: &config,
            data: &data,
            network: &network,
            outcome: &outcome,
            recorder: &recorder,
            started_at: SystemTime::now(),
            training_time: Duration::from_millis(5),
        };
        let dir = write_artifacts(experiment.path(), &run).unwrap();
        assert_eq!(dir, experiment.path().join(ARTIFACTS_DIR));
        assert!(
            !dir.join("grid").exists(),
            "files of the old run are removed"
        );

        let (shape, rows) = read_npy::<u64>(&dir.join("predictions/validation/rows.npy"));
        assert_eq!((shape, rows), (vec![10], (0..10).collect()));
        let (shape, _) = read_npy::<f64>(&dir.join("predictions/test/targets.npy"));
        assert_eq!(shape, [20, 1]);
        assert!(!dir.join("predictions/test/labels.npy").exists());
        // 40 samples in batches of 16: 3 iterations per epoch.
        let (shape, _) = read_npy::<f64>(&dir.join("history/loss.npy"));
        assert_eq!(shape, [6]);

        let metadata: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(dir.join("metadata.json")).unwrap()).unwrap();
        assert_eq!(metadata["grid"], serde_json::Value::Null);
        assert_eq!(metadata["result"]["stop"]["reason"], "completed");
        assert_eq!(metadata["result"]["iterations"], 6);
        assert_eq!(metadata["data"]["samples"]["validation"], 10);
        assert_eq!(
            metadata["network"]["layer_sizes"],
            serde_json::json!([3, 16, 16, 1])
        );
        assert_eq!(metadata["config"]["training"]["epochs"], 2);
        assert_eq!(metadata["training_seconds"], 0.005);
        assert!(metadata["version"]["commit"].is_string());
        assert!(
            metadata["result"]["metrics"]["validation"]["mse"].is_number(),
            "{metadata:#}"
        );
    }
}
