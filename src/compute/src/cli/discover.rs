//! Finding the experiment directories to train.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use miette::Diagnostic;
use thiserror::Error;

use crate::config::CONFIG_FILE_NAME;

/// Directories never searched for experiments: they hold the outputs of a
/// run or of a study.
const SKIPPED_DIRS: [&str; 3] = ["artifacts", "plots", "results"];

/// Failure to find the experiments named on the command line.
#[derive(Debug, Error, Diagnostic)]
pub enum DiscoverError {
    /// The path does not exist or is not a directory.
    #[error("`{}` is not a directory", path.display())]
    #[diagnostic(
        code(train::not_a_directory),
        help("pass an experiment directory or a directory with experiments, e.g. a study")
    )]
    NotADirectory { path: PathBuf },

    /// Neither the directory nor any directory below it has `config.toml`.
    #[error("no `config.toml` in `{}` or below it", path.display())]
    #[diagnostic(
        code(train::no_experiments),
        help("for a study, run `compute generate <study-dir>` first")
    )]
    NoExperiments { path: PathBuf },

    /// A directory cannot be listed.
    #[error("cannot list `{}`", path.display())]
    #[diagnostic(code(train::io))]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
}

/// Expands `dirs` into experiment directories: a directory with
/// `config.toml` is one experiment; any other directory stands for every
/// directory with `config.toml` below it (not looking inside experiments or
/// into `artifacts/`, `plots/` and `results/`), sorted by path. Directories
/// named twice are trained once.
pub(super) fn discover(dirs: &[PathBuf]) -> Result<Vec<PathBuf>, DiscoverError> {
    let mut experiments: Vec<PathBuf> = Vec::new();
    for dir in dirs {
        if !dir.is_dir() {
            return Err(DiscoverError::NotADirectory { path: dir.clone() });
        }
        let mut found = Vec::new();
        search(dir, &mut found)?;
        if found.is_empty() {
            return Err(DiscoverError::NoExperiments { path: dir.clone() });
        }
        found.sort();
        for experiment in found {
            if !experiments.contains(&experiment) {
                experiments.push(experiment);
            }
        }
    }
    Ok(experiments)
}

fn search(dir: &Path, found: &mut Vec<PathBuf>) -> Result<(), DiscoverError> {
    if dir.join(CONFIG_FILE_NAME).is_file() {
        found.push(dir.to_path_buf());
        return Ok(());
    }
    let io_error = |source| DiscoverError::Io {
        path: dir.to_path_buf(),
        source,
    };
    for entry in fs::read_dir(dir).map_err(io_error)? {
        let entry = entry.map_err(io_error)?;
        let skipped = SKIPPED_DIRS.iter().any(|name| entry.file_name() == *name);
        if !skipped && entry.file_type().map_err(io_error)?.is_dir() {
            search(&entry.path(), found)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    /// A tree with experiments at the given paths (relative to its root).
    fn tree(experiments: &[&str]) -> TempDir {
        let root = TempDir::new().unwrap();
        for experiment in experiments {
            let dir = root.path().join(experiment);
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join(CONFIG_FILE_NAME), "").unwrap();
        }
        root
    }

    fn relative(root: &TempDir, dirs: Vec<PathBuf>) -> Vec<PathBuf> {
        dirs.into_iter()
            .map(|dir| dir.strip_prefix(root.path()).unwrap().to_path_buf())
            .collect()
    }

    #[test]
    fn experiment_directory_is_one_run() {
        let root = tree(&["a"]);
        let dirs = discover(&[root.path().join("a")]).unwrap();
        assert_eq!(dirs, [root.path().join("a")]);
    }

    #[test]
    fn finds_experiments_below_a_directory_in_order() {
        let root = tree(&[
            "study/runs/tanh/seed-2",
            "study/runs/relu/seed-1",
            "study/runs/tanh/seed-1",
            "study/runs/tanh/seed-1/artifacts/nested",
            "study/results/fake",
        ]);
        let dirs = discover(&[root.path().join("study")]).unwrap();
        let expected: Vec<PathBuf> = [
            "study/runs/relu/seed-1",
            "study/runs/tanh/seed-1",
            "study/runs/tanh/seed-2",
        ]
        .iter()
        .map(PathBuf::from)
        .collect();
        assert_eq!(relative(&root, dirs), expected);
    }

    #[test]
    fn trains_repeated_directories_once() {
        let root = tree(&["s/a", "s/b"]);
        let dirs = discover(&[root.path().join("s/b"), root.path().join("s")]).unwrap();
        let expected: Vec<PathBuf> = ["s/b", "s/a"].iter().map(PathBuf::from).collect();
        assert_eq!(relative(&root, dirs), expected);
    }

    #[test]
    fn reports_missing_and_empty_directories() {
        let root = tree(&[]);
        fs::create_dir(root.path().join("empty")).unwrap();
        let err = discover(&[root.path().join("missing")]).unwrap_err();
        assert!(matches!(err, DiscoverError::NotADirectory { .. }));
        let err = discover(&[root.path().join("empty")]).unwrap_err();
        assert!(matches!(err, DiscoverError::NoExperiments { .. }));
    }
}
