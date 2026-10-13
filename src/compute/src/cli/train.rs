//! The `train` command.

use std::time::Instant;

use miette::{Severity, miette};

use super::TrainArgs;
use super::progress::Progress;
use super::summary::{DataSummary, NetworkSummary, TrainingSummary, report};
use crate::config::load_experiment;
use crate::data::prepare;
use crate::network::Mlp;
use crate::rng::{RngStream, rng};
use crate::training::{StopReason, train};

/// Loads and checks the experiment config, prepares the data, builds the
/// network and trains it, printing the progress and a final summary.
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

    let mut progress = Progress::new(args.progress_every, config.training.epochs);
    let start = Instant::now();
    let outcome = train(
        &mut network,
        &prepared,
        config.optimizer,
        &config.training,
        &mut progress,
    );
    println!("{}", TrainingSummary(&outcome, start.elapsed()));

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
