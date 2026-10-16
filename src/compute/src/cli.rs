//! Command-line interface of the compute layer.
//!
//! `validate` and `train` take experiment directories that contain
//! `config.toml`; `generate` takes a study directory with `study.toml`.
//! Relative paths, both of the directories and of the datasets named in the
//! configs, are resolved against the current working directory, so commands
//! are meant to be run from the repository root.

mod batch;
mod discover;
mod generate;
mod progress;
mod summary;
mod train;
mod validate;

use std::num::NonZeroUsize;
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
    /// Train the networks described by experiment configs, e.g. all runs of
    /// a study.
    Train(TrainArgs),
    /// Write the experiment directories of a study described by `study.toml`.
    Generate(StudyArgs),
}

/// Arguments shared by the commands that operate on one experiment.
#[derive(Debug, Args)]
pub struct ExperimentArgs {
    /// Experiment directory containing `config.toml`, e.g.
    /// `data/experiments/classification/<name>`.
    pub experiment_dir: PathBuf,
}

/// Arguments of the commands that operate on a study.
#[derive(Debug, Args)]
pub struct StudyArgs {
    /// Study directory containing `study.toml`, e.g.
    /// `data/experiments/classification/<study>`.
    pub study_dir: PathBuf,
}

/// Arguments of the `train` command.
#[derive(Debug, Args)]
pub struct TrainArgs {
    /// Experiment directories containing `config.toml`, or directories
    /// searched for them, e.g. a study directory; several runs are trained
    /// in parallel.
    #[arg(required = true, value_name = "DIR")]
    pub experiment_dirs: Vec<PathBuf>,
    /// Print the metrics every N epochs, before training and after the last
    /// epoch; 0 prints only the final summary. Defaults to about 20 lines.
    /// Only for a single run: several runs print one line each.
    #[arg(long, value_name = "N")]
    pub progress_every: Option<usize>,
    /// Train without recording and writing artifacts, e.g. to measure the
    /// logging overhead. Existing artifacts are left untouched.
    #[arg(long)]
    pub no_artifacts: bool,
    /// Number of runs trained at the same time. Defaults to the number of
    /// CPUs.
    #[arg(long, short, value_name = "N")]
    pub jobs: Option<NonZeroUsize>,
    /// Skip runs whose artifacts were written from the same config, e.g.
    /// after adding seeds or levels to a study.
    #[arg(long)]
    pub skip_trained: bool,
}

/// Runs the parsed command.
pub fn run(cli: Cli) -> miette::Result<()> {
    match cli.command {
        Command::Validate(args) => validate::run(&args),
        Command::Train(args) => train::run(&args),
        Command::Generate(args) => generate::run(&args),
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
        assert_eq!(args.experiment_dirs, [PathBuf::from("experiment")]);
        assert_eq!(args.progress_every, None);
        assert_eq!(args.jobs, None);
        assert!(!args.no_artifacts);
        assert!(!args.skip_trained);
        let args = parse(&[
            "compute",
            "train",
            "experiment",
            "--progress-every",
            "0",
            "--no-artifacts",
        ]);
        assert_eq!(args.progress_every, Some(0));
        assert!(args.no_artifacts);
        let args = parse(&["compute", "train", "a", "b", "-j", "4", "--skip-trained"]);
        assert_eq!(args.experiment_dirs.len(), 2);
        assert_eq!(args.jobs, NonZeroUsize::new(4));
        assert!(args.skip_trained);
        assert!(Cli::try_parse_from(["compute", "train", "a", "--jobs", "0"]).is_err());
    }

    #[test]
    fn parses_study_dir() {
        let cli = Cli::try_parse_from(["compute", "generate", "data/experiments/x"]).unwrap();
        let Command::Generate(args) = cli.command else {
            panic!("expected the generate command");
        };
        assert_eq!(args.study_dir, PathBuf::from("data/experiments/x"));
    }

    #[test]
    fn requires_command_and_experiment_dir() {
        assert!(Cli::try_parse_from(["compute"]).is_err());
        assert!(Cli::try_parse_from(["compute", "train"]).is_err());
    }
}
