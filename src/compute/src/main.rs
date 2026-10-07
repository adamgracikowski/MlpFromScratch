//! Command-line entry point of the compute layer.

use clap::Parser;

/// Train a multilayer perceptron and write training artifacts.
#[derive(Debug, Parser)]
#[command(name = "compute", version, about)]
struct Cli {}

fn main() -> miette::Result<()> {
    let _cli = Cli::parse();
    Ok(())
}
