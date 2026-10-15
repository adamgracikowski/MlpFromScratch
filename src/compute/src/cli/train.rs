//! The `train` command.

use std::fs;
use std::path::Path;
use std::time::{Instant, SystemTime};

use miette::{Severity, miette};

use super::TrainArgs;
use super::batch;
use super::discover::discover;
use super::progress::Progress;
use super::summary::{DataSummary, NetworkSummary, TrainingSummary, report};
use crate::artifacts::{ARTIFACTS_DIR, Recorder, Run, write_artifacts};
use crate::config::{ExperimentConfig, LoadedConfig, load_experiment};
use crate::data::prepare;
use crate::network::Mlp;
use crate::rng::{RngStream, rng};
use crate::training::{StopReason, TrainingOutcome, train};

/// Trains the experiments named on the command line. A single experiment
/// prints its config, progress and results; several are trained in parallel
/// with one line per finished run (see [`batch::run`]).
pub(super) fn run(args: &TrainArgs) -> miette::Result<()> {
    let dirs = discover(&args.experiment_dirs)?;
    let options = TrainOptions {
        progress_every: args.progress_every,
        no_artifacts: args.no_artifacts,
        skip_trained: args.skip_trained,
    };
    match dirs.as_slice() {
        [dir] => train_verbose(dir, &options),
        _ => batch::run(&dirs, &args.experiment_dirs, &options, args.jobs),
    }
}

/// Settings shared by all runs of one `train` command.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct TrainOptions {
    /// Progress interval in epochs of a single verbose run.
    pub progress_every: Option<usize>,
    /// Train without recording and writing artifacts.
    pub no_artifacts: bool,
    /// Skip runs whose artifacts were written from the same config.
    pub skip_trained: bool,
}

/// What happened to one experiment directory.
#[derive(Debug)]
pub(super) enum RunResult {
    /// The network was trained (and the artifacts written, unless disabled).
    Trained(Box<TrainingOutcome>),
    /// The artifacts already match the config (`--skip-trained`).
    Skipped,
}

/// Loads and checks the experiment config, prepares the data, builds the
/// network and trains it, printing the progress and a final summary. Unless
/// `--no-artifacts` is given, the training history, the trained network and
/// its predictions are then written to `<experiment-dir>/artifacts/`.
///
/// A diverged run is reported as a warning, not as an error: it is a valid
/// result of an experiment (e.g. too large a learning rate).
fn train_verbose(dir: &Path, options: &TrainOptions) -> miette::Result<()> {
    let loaded = load_experiment(dir)?;
    if options.skip_trained && is_trained(dir, &loaded.config) {
        println!(
            "skipped:    {} is already trained with this config",
            dir.display()
        );
        return Ok(());
    }
    report(&loaded);
    let outcome = train_loaded(dir, &loaded, options, true)?;
    if let StopReason::Diverged { epoch, .. } = outcome.stop {
        let warning = miette!(
            severity = Severity::Warning,
            help = "try a smaller optimizer.learning_rate",
            "training diverged in epoch {epoch}: the training loss is not finite"
        );
        eprintln!("{warning:?}");
    }
    Ok(())
}

/// Trains one experiment of a batch without printing anything; config
/// warnings are not repeated either.
pub(super) fn train_quiet(dir: &Path, options: &TrainOptions) -> miette::Result<RunResult> {
    let loaded = load_experiment(dir)?;
    if options.skip_trained && is_trained(dir, &loaded.config) {
        return Ok(RunResult::Skipped);
    }
    let quiet = TrainOptions {
        progress_every: Some(0),
        ..*options
    };
    let outcome = train_loaded(dir, &loaded, &quiet, false)?;
    Ok(RunResult::Trained(Box::new(outcome)))
}

/// Prepares the data, trains the network and writes the artifacts; with
/// `verbose`, prints the data, network and training summaries.
fn train_loaded(
    dir: &Path,
    loaded: &LoadedConfig,
    options: &TrainOptions,
    verbose: bool,
) -> miette::Result<TrainingOutcome> {
    let prepared = prepare(loaded)?;
    if verbose {
        println!("{}", DataSummary(&prepared));
    }

    let config = &loaded.config;
    let mut network = Mlp::new(
        &config.model,
        prepared.train.inputs.ncols(),
        prepared.train.targets.ncols(),
        &mut rng(config.training.seed, RngStream::Init),
    );
    if verbose {
        println!("{}", NetworkSummary(&network));
    }

    let progress = Progress::new(options.progress_every, config.training.epochs);
    let recorder = (!options.no_artifacts)
        .then(|| Recorder::new(&network, &config.logging, config.training.seed));
    let mut observer = (progress, recorder);
    let started_at = SystemTime::now();
    let start = Instant::now();
    let outcome = train(
        &mut network,
        &prepared,
        config.optimizer,
        &config.training,
        &mut observer,
    );
    let training_time = start.elapsed();
    if verbose {
        println!("{}", TrainingSummary(&outcome, training_time));
    }

    if let (_, Some(recorder)) = &observer {
        let start = Instant::now();
        let run = Run {
            config,
            data: &prepared,
            network: &network,
            outcome: &outcome,
            recorder,
            started_at,
            training_time,
        };
        let artifacts = write_artifacts(dir, &run)?;
        if verbose {
            println!(
                "artifacts:  {} in {:.2} s",
                artifacts.display(),
                start.elapsed().as_secs_f64()
            );
        }
    }
    Ok(outcome)
}

/// Whether `dir/artifacts/metadata.json` exists and was written from
/// `config`. The metadata is written last, so it marks complete artifacts;
/// a diverged run counts as trained. Unreadable metadata counts as not
/// trained.
pub(super) fn is_trained(dir: &Path, config: &ExperimentConfig) -> bool {
    let path = dir.join(ARTIFACTS_DIR).join("metadata.json");
    let Ok(text) = fs::read_to_string(path) else {
        return false;
    };
    let Ok(mut metadata) = serde_json::from_str::<serde_json::Value>(&text) else {
        return false;
    };
    serde_json::from_value::<ExperimentConfig>(metadata["config"].take())
        .is_ok_and(|trained| trained == *config)
}
