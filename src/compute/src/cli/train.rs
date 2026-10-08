//! The `train` command.

use miette::miette;

use super::ExperimentArgs;
use super::summary::{DataSummary, NetworkSummary, report};
use crate::config::load_experiment;
use crate::data::prepare;
use crate::network::Mlp;
use crate::rng::{RngStream, rng};

/// Loads and checks the experiment config, prepares the data, builds the
/// network, then trains it.
///
/// Training itself is not implemented yet, so the command stops with an
/// error after the network is built.
pub(super) fn run(args: &ExperimentArgs) -> miette::Result<()> {
    let loaded = load_experiment(&args.experiment_dir)?;
    report(&loaded);
    let prepared = prepare(&loaded)?;
    println!("{}", DataSummary(&prepared));

    let config = &loaded.config;
    let network = Mlp::new(
        &config.model,
        prepared.train.inputs.ncols(),
        prepared.train.targets.ncols(),
        &mut rng(config.training.seed, RngStream::Init),
    );
    println!("{}", NetworkSummary(&network));
    Err(miette!(
        help = "use `compute validate` to check a config",
        "training is not implemented yet"
    ))
}
