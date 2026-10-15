//! The `train` command.

use std::time::{Instant, SystemTime};

use miette::{Severity, miette};

use super::TrainArgs;
use super::progress::Progress;
use super::summary::{DataSummary, NetworkSummary, TrainingSummary, report};
use crate::artifacts::{Recorder, Run, write_artifacts};
use crate::config::load_experiment;
use crate::data::prepare;
use crate::network::Mlp;
use crate::rng::{RngStream, rng};
use crate::training::{StopReason, train};

/// Loads and checks the experiment config, prepares the data, builds the
/// network and trains it, printing the progress and a final summary. Unless
/// `--no-artifacts` is given, the training history, the trained network and
/// its predictions are then written to `<experiment-dir>/artifacts/`.
///
/// A diverged run is reported as a warning, not as an error: it is a valid
/// result of an experiment (e.g. too large a learning rate).
pub(super) fn run(args: &TrainArgs) -> miette::Result<()> {
    let loaded = load_experiment(&args.experiment.experiment_dir)?;
    report(&loaded);
    let prepared = prepare(&loaded)?;
    println!("{}", DataSummary(&prepared));

    let config = &loaded.config;
    let mut network = Mlp::new(
        &config.model,
        prepared.train.inputs.ncols(),
        prepared.train.targets.ncols(),
        &mut rng(config.training.seed, RngStream::Init),
    );
    println!("{}", NetworkSummary(&network));

    let progress = Progress::new(args.progress_every, config.training.epochs);
    let recorder = (!args.no_artifacts)
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
    println!("{}", TrainingSummary(&outcome, training_time));

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
        let dir = write_artifacts(&args.experiment.experiment_dir, &run)?;
        println!(
            "artifacts:  {} in {:.2} s",
            dir.display(),
            start.elapsed().as_secs_f64()
        );
    }

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
