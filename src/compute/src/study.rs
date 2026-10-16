//! Studies: grids of experiments described by one `study.toml`.
//!
//! A study holds a complete base config (`[base]`, without `training.seed`),
//! a list of seeds and a list of axes. Every axis has named levels, each of
//! which overrides one or more config keys. The cartesian product of all
//! levels is the set of variants; every variant runs once per seed. Each run
//! is an ordinary experiment directory,
//! `<study>/runs/<level>__<level>/seed-<seed>/config.toml`, so `train` and the
//! plotting scripts work on it unchanged. `runs/index.csv` maps every run to
//! its variant, seed and axis levels.

mod axis;
mod error;
#[cfg(test)]
mod examples;
mod expand;
#[cfg(test)]
mod fixtures;
mod load;
mod study_config;
mod validate;
mod write;

pub use axis::{Axis, Level, LevelConfig, level_name};
pub use error::{RunError, StudyError};
pub use expand::{RunSpec, expand};
pub use load::{GeneratedRun, Study, load_study};
pub use study_config::{ReportConfig, StudyConfig};
pub use validate::validate;
pub use write::{WriteReport, write_runs};

/// Name of the study file inside a study directory.
pub const STUDY_FILE_NAME: &str = "study.toml";

/// Directory of the generated runs inside a study directory.
pub const RUNS_DIR: &str = "runs";

/// Index of the generated runs inside [`RUNS_DIR`].
pub const INDEX_FILE_NAME: &str = "index.csv";

/// Variant name of a study without axes.
pub const BASE_VARIANT: &str = "base";

/// Separator of the level names in a variant name.
pub const LEVEL_SEPARATOR: &str = "__";
