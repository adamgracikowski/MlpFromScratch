//! The `generate` command.

use std::collections::HashSet;
use std::fmt;

use super::StudyArgs;
use crate::study::{Study, WriteReport, load_study, write_runs};

/// Loads and checks the study, then writes the config of every run, removes
/// runs that the study no longer generates and writes `runs/index.csv`.
/// Nothing is written when any generated config is invalid.
pub(super) fn run(args: &StudyArgs) -> miette::Result<()> {
    let study = load_study(&args.study_dir)?;
    // Warnings do not depend on the seed: report them once per variant.
    let mut reported = HashSet::new();
    for run in &study.runs {
        if reported.insert(&run.spec.variant) {
            for warning in &run.warnings {
                eprintln!("{:?}", miette::Report::new(warning.clone()));
            }
        }
    }
    let report = write_runs(&study)?;
    print!("{}", GenerateSummary(&study, &report));
    Ok(())
}

/// Summary of a generated study: axes, seeds, runs and changes on disk.
struct GenerateSummary<'a>(&'a Study, &'a WriteReport);

impl fmt::Display for GenerateSummary<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let GenerateSummary(study, report) = self;
        let config = &study.config;
        writeln!(f, "study:      {}", study.dir.display())?;
        let axes: Vec<String> = config
            .axes
            .iter()
            .map(|axis| format!("{} ({})", axis.name, axis.levels().len()))
            .collect();
        if axes.is_empty() {
            writeln!(f, "axes:       none")?;
        } else {
            writeln!(f, "axes:       {}", axes.join(", "))?;
        }
        let seeds: Vec<String> = config.seeds.iter().map(u64::to_string).collect();
        writeln!(f, "seeds:      {}", seeds.join(", "))?;
        let variants = study.runs.len() / config.seeds.len();
        writeln!(
            f,
            "runs:       {} ({variants} variant(s) x {} seed(s))",
            study.runs.len(),
            config.seeds.len()
        )?;
        writeln!(
            f,
            "configs:    {} written, {} unchanged",
            report.written, report.unchanged
        )?;
        for removed in &report.removed {
            writeln!(f, "removed:    {}", removed.display())?;
        }
        writeln!(f, "index:      {}", report.index.display())
    }
}
