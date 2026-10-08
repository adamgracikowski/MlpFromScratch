//! The `train` command.

use miette::miette;

use super::ExperimentArgs;
use super::summary::report;
use crate::config::load_experiment;

/// Loads and checks the experiment config, then trains the network.
///
/// Training itself is not implemented yet, so the command stops with an
/// error after the config checks.
pub(super) fn run(args: &ExperimentArgs) -> miette::Result<()> {
    let loaded = load_experiment(&args.experiment_dir)?;
    report(&loaded);
    Err(miette!(
        help = "use `compute validate` to check a config",
        "training is not implemented yet"
    ))
}
