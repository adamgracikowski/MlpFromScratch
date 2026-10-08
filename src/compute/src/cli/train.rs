//! The `train` command.

use miette::miette;

use super::ExperimentArgs;
use super::summary::{DataSummary, report};
use crate::config::load_experiment;
use crate::data::prepare;

/// Loads and checks the experiment config, prepares the data, then trains
/// the network.
///
/// Training itself is not implemented yet, so the command stops with an
/// error after the data is prepared.
pub(super) fn run(args: &ExperimentArgs) -> miette::Result<()> {
    let loaded = load_experiment(&args.experiment_dir)?;
    report(&loaded);
    let prepared = prepare(&loaded)?;
    println!("{}", DataSummary(&prepared));
    Err(miette!(
        help = "use `compute validate` to check a config",
        "training is not implemented yet"
    ))
}
