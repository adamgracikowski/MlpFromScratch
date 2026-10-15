//! `metadata.json`: what was run, with which code, and how it ended.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::UNIX_EPOCH;

use serde::Serialize;

use super::{ArtifactError, Grid, Run};
use crate::config::{ExperimentConfig, Normalization};
use crate::data::Scaler;
use crate::training::{SplitMetrics, StopReason};

/// Contents of `metadata.json`.
#[derive(Debug, Serialize)]
pub(super) struct Metadata<'a> {
    version: Version,
    /// Start of training, in seconds since the Unix epoch.
    started_at: u64,
    /// Wall time of the training loop, without loading data and writing
    /// artifacts.
    training_seconds: f64,
    /// The full config, with defaults filled in.
    config: &'a ExperimentConfig,
    data: DataInfo<'a>,
    network: NetworkInfo,
    result: ResultInfo,
    history: HistoryInfo,
    /// Absent when the inputs have more than two features.
    grid: Option<GridInfo>,
}

/// Version of the code that produced the artifacts.
#[derive(Debug, Serialize)]
struct Version {
    /// Version of the `compute` crate.
    package: &'static str,
    /// Git commit the binary was built from (`unknown` without git).
    commit: &'static str,
    /// Whether the crate sources differed from that commit; `null` if unknown.
    dirty: Option<bool>,
}

impl Version {
    fn current() -> Self {
        Self {
            package: env!("CARGO_PKG_VERSION"),
            commit: env!("COMPUTE_GIT_COMMIT"),
            dirty: match env!("COMPUTE_GIT_DIRTY") {
                "true" => Some(true),
                "false" => Some(false),
                _ => None,
            },
        }
    }
}

#[derive(Debug, Serialize)]
struct DataInfo<'a> {
    features: &'a [String],
    target: &'a str,
    samples: Samples,
    /// Class labels behind the network outputs (classification only).
    classes: Option<&'a [i64]>,
    input_scaler: ScalerInfo,
    /// Normalization of the target (regression only).
    target_scaler: Option<ScalerInfo>,
}

#[derive(Debug, Serialize)]
struct Samples {
    train: usize,
    validation: Option<usize>,
    test: usize,
}

/// Normalization `(x - offset) / scale` of every column.
#[derive(Debug, Serialize)]
struct ScalerInfo {
    normalization: Normalization,
    offset: Vec<f64>,
    scale: Vec<f64>,
}

impl From<&Scaler> for ScalerInfo {
    fn from(scaler: &Scaler) -> Self {
        Self {
            normalization: scaler.normalization,
            offset: scaler.offset.to_vec(),
            scale: scaler.scale.to_vec(),
        }
    }
}

#[derive(Debug, Serialize)]
struct NetworkInfo {
    /// Sizes from the input to the output layer, e.g. `[2, 16, 16, 3]`.
    layer_sizes: Vec<usize>,
    /// Activation of every layer, e.g. `["tanh", "tanh", "softmax"]`.
    activations: Vec<String>,
    bias: bool,
    parameter_count: usize,
}

#[derive(Debug, Serialize)]
struct ResultInfo {
    stop: StopReason,
    epochs: usize,
    iterations: usize,
    /// With early stopping, the epoch of the saved network.
    best_epoch: Option<usize>,
    /// Metrics of the saved network.
    metrics: SplitMetrics,
}

#[derive(Debug, Serialize)]
struct HistoryInfo {
    loss_every: usize,
    loss_points: usize,
    weights_every: usize,
    snapshots: usize,
    /// Number of recorded parameters of every layer.
    sampled_parameters: Vec<usize>,
}

#[derive(Debug, Serialize)]
struct GridInfo {
    resolution: usize,
    /// `[low, high]` of every axis.
    ranges: Vec<[f64; 2]>,
}

impl<'a> Metadata<'a> {
    /// Collects the metadata of `run`, whose grid (if any) is `grid`.
    pub(super) fn new(run: &Run<'a>, grid: Option<&Grid>) -> Self {
        let Run {
            config,
            data,
            network,
            outcome,
            recorder,
            ..
        } = *run;
        let layers = network.layers();
        let mut layer_sizes = vec![network.input_size()];
        layer_sizes.extend(layers.iter().map(|layer| layer.output_size()));
        Self {
            version: Version::current(),
            started_at: run
                .started_at
                .duration_since(UNIX_EPOCH)
                .map_or(0, |since| since.as_secs()),
            training_seconds: run.training_time.as_secs_f64(),
            config,
            data: DataInfo {
                features: &data.feature_names,
                target: &config.data.target,
                samples: Samples {
                    train: data.train.len(),
                    validation: data.validation.as_ref().map(|dataset| dataset.len()),
                    test: data.test.len(),
                },
                classes: data.classes.as_ref().map(|classes| classes.classes()),
                input_scaler: ScalerInfo::from(&data.input_scaler),
                target_scaler: data.target_scaler.as_ref().map(ScalerInfo::from),
            },
            network: NetworkInfo {
                layer_sizes,
                activations: layers
                    .iter()
                    .map(|layer| layer.activation().to_string())
                    .collect(),
                bias: config.model.bias,
                parameter_count: network.parameter_count(),
            },
            result: ResultInfo {
                stop: outcome.stop,
                epochs: outcome.epochs,
                iterations: outcome.iterations,
                best_epoch: outcome.best_epoch,
                metrics: outcome.metrics,
            },
            history: HistoryInfo {
                loss_every: config.logging.loss_every,
                loss_points: recorder.loss().losses().len(),
                weights_every: config.logging.weights_every,
                snapshots: recorder.weights().iterations().len(),
                sampled_parameters: recorder
                    .weights()
                    .sample()
                    .layers()
                    .iter()
                    .map(Vec::len)
                    .collect(),
            },
            grid: grid.map(|grid| GridInfo {
                resolution: config.output.grid_resolution,
                ranges: grid
                    .axes
                    .iter()
                    .map(|axis| [axis[0], axis[axis.len() - 1]])
                    .collect(),
            }),
        }
    }

    /// Writes the metadata as pretty-printed JSON; NaN and infinite numbers
    /// (after divergence) become `null`.
    pub(super) fn write(&self, path: &Path) -> Result<(), ArtifactError> {
        let file = File::create(path).map_err(|source| ArtifactError::write(path, source))?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, self)
            .map_err(|source| ArtifactError::write(path, source))?;
        writer
            .write_all(b"\n")
            .and_then(|()| writer.flush())
            .map_err(|source| ArtifactError::write(path, source))
    }
}
