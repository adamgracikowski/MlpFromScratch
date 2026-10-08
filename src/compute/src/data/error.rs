//! Errors of loading and preparing datasets.

use std::path::PathBuf;

use miette::Diagnostic;
use thiserror::Error;

/// Failure to load or prepare a dataset.
#[derive(Debug, Error, Diagnostic)]
pub enum DataError {
    /// The file cannot be opened or is not well-formed CSV.
    #[error("cannot read `{}`", path.display())]
    #[diagnostic(
        code(data::read),
        help("paths are resolved relative to the current working directory")
    )]
    Read {
        path: PathBuf,
        #[source]
        source: csv::Error,
    },

    /// The file has no header row.
    #[error("`{}` is empty", path.display())]
    #[diagnostic(code(data::empty_file))]
    EmptyFile { path: PathBuf },

    /// The target column is not in the header.
    #[error("column `{column}` not found in `{}`", path.display())]
    #[diagnostic(code(data::missing_column), help("available columns: {available}"))]
    MissingColumn {
        path: PathBuf,
        column: String,
        available: String,
    },

    /// The header has only the target column.
    #[error("`{}` has no feature columns", path.display())]
    #[diagnostic(code(data::no_features))]
    NoFeatures { path: PathBuf },

    /// The file has a header but no samples.
    #[error("`{}` has no data rows", path.display())]
    #[diagnostic(code(data::no_rows))]
    NoRows { path: PathBuf },

    /// A field is not a finite number.
    #[error("{}:{line}: invalid value `{value}` in column `{column}`", path.display())]
    #[diagnostic(code(data::invalid_value), help("every value must be a finite number"))]
    InvalidValue {
        path: PathBuf,
        line: u64,
        column: String,
        value: String,
    },

    /// A class label is not an integer.
    #[error("{}:{line}: class label `{value}` is not an integer", path.display())]
    #[diagnostic(
        code(data::non_integer_label),
        help("classification targets must be integer labels")
    )]
    NonIntegerLabel {
        path: PathBuf,
        line: u64,
        value: f64,
    },

    /// The training set has fewer than two classes.
    #[error("`{}` has fewer than two classes (found: {classes})", path.display())]
    #[diagnostic(
        code(data::too_few_classes),
        help("classification needs at least two classes")
    )]
    TooFewClasses { path: PathBuf, classes: String },

    /// A label of the test set does not occur in the training set.
    #[error("{}:{line}: class label `{label}` does not occur in the training set", path.display())]
    #[diagnostic(code(data::unknown_label), help("training classes: {classes}"))]
    UnknownLabel {
        path: PathBuf,
        line: u64,
        label: i64,
        classes: String,
    },
}
