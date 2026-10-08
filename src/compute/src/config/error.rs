//! Errors of loading an experiment config.

use std::io;
use std::path::PathBuf;

use miette::{Diagnostic, NamedSource, SourceSpan};
use thiserror::Error;

use super::IssueDiagnostic;

/// Failure to load an experiment config.
#[derive(Debug, Error, Diagnostic)]
pub enum ConfigError {
    /// The config file cannot be read.
    #[error("cannot read config `{}`", path.display())]
    #[diagnostic(
        code(config::io),
        help("pass an experiment directory that contains `config.toml`")
    )]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    /// The file is not valid TOML or does not match the schema.
    #[error("cannot parse config: {message}")]
    #[diagnostic(code(config::parse))]
    Parse {
        message: String,
        #[source_code]
        src: NamedSource<String>,
        #[label("here")]
        span: Option<SourceSpan>,
    },

    /// The config parsed, but some values are invalid.
    #[error("config `{}` has {error_count} invalid value(s)", path.display())]
    #[diagnostic(code(config::invalid))]
    Invalid {
        path: PathBuf,
        error_count: usize,
        /// All issues found, including warnings.
        #[related]
        issues: Vec<IssueDiagnostic>,
    },
}
