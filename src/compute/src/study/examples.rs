//! Checks that every study in `data/experiments/` is valid and that every config it
//! generates is valid, with existing dataset files.

use std::fs;
use std::path::{Path, PathBuf};

use miette::Severity;

use super::{STUDY_FILE_NAME, StudyConfig, expand, validate};
use crate::config::{ExperimentConfig, Issue, check_files};

/// Datasets that are downloaded instead of committed (see the README in this directory).
/// Their files are checked only when present, e.g. not in CI.
const DOWNLOADED_DATASETS: &str = "data/datasets/mnist";

/// The repository root; dataset paths in studies are relative to it.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Directories `data/experiments/<task>/<study>/` with a `study.toml`, sorted.
fn studies(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for task in fs::read_dir(root.join("data/experiments")).unwrap() {
        let task = task.unwrap().path();
        if !task.is_dir() {
            continue;
        }
        for study in fs::read_dir(&task).unwrap() {
            let study = study.unwrap().path();
            if study.join(STUDY_FILE_NAME).is_file() {
                found.push(study);
            }
        }
    }
    found.sort();
    found
}

/// Errors and warnings; advice (e.g. the unused hidden activation of a network without
/// hidden layers) is expected in some variants.
fn problems(issues: &[Issue]) -> Vec<String> {
    issues
        .iter()
        .filter(|issue| issue.severity != Severity::Advice)
        .map(Issue::to_string)
        .collect()
}

#[test]
fn studies_generate_valid_configs() {
    let root = repo_root();
    let studies = studies(&root);
    assert!(studies.len() >= 14, "found only {studies:?}");
    for study_dir in studies {
        let text = fs::read_to_string(study_dir.join(STUDY_FILE_NAME)).unwrap();
        let study: StudyConfig = toml::from_str(&text).unwrap();
        let name = study_dir.display();
        let issues = problems(&validate(&study));
        assert!(issues.is_empty(), "{name}: {issues:?}");

        for run in expand(&study) {
            let name = format!("{name}/{}", run.dir.display());
            let text = toml::to_string(&run.config).unwrap();
            let mut config: ExperimentConfig =
                toml::from_str(&text).unwrap_or_else(|err| panic!("{name}: {err}"));
            let missing: Vec<PathBuf> = [&config.data.train, &config.data.test]
                .into_iter()
                .filter(|path| !root.join(path).is_file())
                .cloned()
                .collect();
            let downloaded = missing
                .iter()
                .all(|path| path.starts_with(DOWNLOADED_DATASETS));
            // Tests run in the crate directory, not in the repository root.
            config.data.train = root.join(&config.data.train);
            config.data.test = root.join(&config.data.test);
            let mut issues = crate::config::validate(&config);
            if missing.is_empty() || !downloaded {
                issues.extend(check_files(&config.data).issues);
            }
            let issues = problems(&issues);
            assert!(issues.is_empty(), "{name}: {issues:?}");
        }
    }
}
