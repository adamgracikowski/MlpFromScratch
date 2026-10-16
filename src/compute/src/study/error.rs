//! Errors of loading a study and writing its runs.

use std::io;
use std::path::PathBuf;

use miette::{Diagnostic, NamedSource, SourceSpan};
use thiserror::Error;

use crate::config::{ConfigError, IssueDiagnostic};

/// Failure to load a study or to write its runs.
#[derive(Debug, Error, Diagnostic)]
pub enum StudyError {
    /// The study file cannot be read.
    #[error("cannot read study `{}`", path.display())]
    #[diagnostic(
        code(study::io),
        help("pass a study directory that contains `study.toml`")
    )]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    /// The file is not valid TOML or does not match the schema.
    #[error("cannot parse study: {message}")]
    #[diagnostic(code(study::parse))]
    Parse {
        message: String,
        #[source_code]
        src: NamedSource<String>,
        #[label("here")]
        span: Option<SourceSpan>,
    },

    /// The study parsed, but some values are invalid.
    #[error("study `{}` has {error_count} invalid value(s)", path.display())]
    #[diagnostic(code(study::invalid))]
    Invalid {
        path: PathBuf,
        error_count: usize,
        #[related]
        issues: Vec<IssueDiagnostic>,
    },

    /// The configs of some variants are invalid.
    #[error(
        "{} variant(s) of the study have an invalid config",
        invalid_variants(errors)
    )]
    #[diagnostic(
        code(study::runs),
        help("fix `[base]` or the levels in `study.toml`; nothing was written")
    )]
    Runs {
        #[related]
        errors: Vec<RunError>,
    },

    /// A run directory or file cannot be written or removed.
    #[error("cannot write `{}`", path.display())]
    #[diagnostic(code(study::write))]
    Write {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
}

/// The generated config of one run is invalid. Only the first seed of each
/// variant is checked, since the seed does not change the other values, and
/// variants with the same issues (e.g. an invalid `[base]`) are reported once.
#[derive(Debug, Error, Diagnostic)]
#[error("variant `{variant}` (seed {seed}) has an invalid config{}", same_in_suffix(.same_in))]
pub struct RunError {
    /// Name of the variant.
    pub variant: String,
    /// Seed of the reported run.
    pub seed: u64,
    /// Other variants whose configs have the same issues.
    pub same_in: Vec<String>,
    /// The error of loading the generated config, pointing into its text.
    #[diagnostic_source]
    pub config_error: ConfigError,
}

fn invalid_variants(errors: &[RunError]) -> usize {
    errors.iter().map(|error| 1 + error.same_in.len()).sum()
}

fn same_in_suffix(same_in: &[String]) -> String {
    match same_in.len() {
        0 => String::new(),
        count => format!(" (the same in {count} other variant(s))"),
    }
}
