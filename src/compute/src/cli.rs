//! Command-line interface of the compute layer.
//!
//! Every command takes an experiment directory that contains `config.toml`.
//! Relative paths, both of the directory and of the datasets named in the
//! config, are resolved against the current working directory, so commands
//! are meant to be run from the repository root.

mod progress;
mod summary;
mod train;
mod validate;

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

/// Train a multilayer perceptron and write training artifacts.
#[derive(Debug, Parser)]
#[command(name = "compute", version, about)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

/// Commands of the compute binary.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Check an experiment config and its dataset files without training.
    Validate(ExperimentArgs),
    /// Train the network described by an experiment config.
    Train(TrainArgs),
}

/// Arguments shared by the commands that operate on one experiment.
#[derive(Debug, Args)]
pub struct ExperimentArgs {
    /// Experiment directory containing `config.toml`, e.g.
    /// `data/experiments/classification/<name>`.
    pub experiment_dir: PathBuf,
}

/// Arguments of the `train` command.
#[derive(Debug, Args)]
pub struct TrainArgs {
    #[command(flatten)]
    pub experiment: ExperimentArgs,
    /// Print the metrics every N epochs, before training and after the last
    /// epoch; 0 prints only the final summary. Defaults to about 20 lines.
    #[arg(long, value_name = "N")]
    pub progress_every: Option<usize>,
}

/// Runs the parsed command.
pub fn run(cli: Cli) -> miette::Result<()> {
    match cli.command {
        Command::Validate(args) => validate::run(&args),
        Command::Train(args) => train::run(&args),
    }
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    use super::*;

    #[test]
    fn command_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn parses_experiment_dir() {
        let cli = Cli::try_parse_from(["compute", "validate", "data/experiments/x"]).unwrap();
        let Command::Validate(args) = cli.command else {
            panic!("expected the validate command");
        };
        assert_eq!(args.experiment_dir, PathBuf::from("data/experiments/x"));
    }

    #[test]
    fn parses_train_progress_interval() {
        let parse = |args: &[&str]| {
            let cli = Cli::try_parse_from(args).unwrap();
            let Command::Train(args) = cli.command else {
                panic!("expected the train command");
            };
            args
        };
        let args = parse(&["compute", "train", "experiment"]);
        assert_eq!(args.experiment.experiment_dir, PathBuf::from("experiment"));
        assert_eq!(args.progress_every, None);
        let args = parse(&["compute", "train", "experiment", "--progress-every", "0"]);
        assert_eq!(args.progress_every, Some(0));
    }

    #[test]
    fn requires_command_and_experiment_dir() {
        assert!(Cli::try_parse_from(["compute"]).is_err());
        assert!(Cli::try_parse_from(["compute", "train"]).is_err());
    }
}
