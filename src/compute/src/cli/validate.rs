//! The `validate` command.

use super::ExperimentArgs;
use super::summary::report;
use crate::config::load_experiment;

/// Loads and checks the experiment config, then prints its summary.
pub(super) fn run(args: &ExperimentArgs) -> miette::Result<()> {
    let loaded = load_experiment(&args.experiment_dir)?;
    report(&loaded);
    Ok(())
}
