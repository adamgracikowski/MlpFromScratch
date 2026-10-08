//! Command-line entry point of the compute layer.

use clap::Parser;
use compute::cli::{Cli, run};

fn main() -> miette::Result<()> {
    run(Cli::parse())
}
