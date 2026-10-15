//! Errors of writing the artifacts to disk.

use std::io;
use std::path::PathBuf;

use miette::Diagnostic;
use thiserror::Error;

/// A file or directory of the artifacts could not be written.
#[derive(Debug, Error, Diagnostic)]
pub enum ArtifactError {
    /// The artifacts of a previous run could not be removed.
    #[error("cannot remove the previous artifacts in `{}`", path.display())]
    #[diagnostic(
        code(artifacts::clear),
        help("close programs that use files in this directory, or delete it by hand")
    )]
    Clear {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    /// A directory or file could not be created or written.
    #[error("cannot write `{}`", path.display())]
    #[diagnostic(
        code(artifacts::write),
        help("check that the experiment directory is writable and the disk is not full")
    )]
    Write {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
}

impl ArtifactError {
    /// Wraps a failure to write `path`.
    pub(super) fn write(path: impl Into<PathBuf>, source: impl Into<io::Error>) -> Self {
        Self::Write {
            path: path.into(),
            source: source.into(),
        }
    }
}
