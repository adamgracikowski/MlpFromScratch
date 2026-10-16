//! Checks that the example studies in `data/experiments/` are valid and that
//! every config they generate is valid, with existing dataset files.

use std::fs;
use std::path::{Path, PathBuf};

use super::{STUDY_FILE_NAME, StudyConfig, expand, validate};
use crate::config::{ExperimentConfig, Issue, check_files};

/// Example studies, relative to the repository root.
const EXAMPLES: [&str; 2] = [
    "data/experiments/classification/example-study",
    "data/experiments/regression/example-study",
];

/// The repository root; dataset paths in studies are relative to it.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn example_studies_generate_valid_configs() {
    let root = repo_root();
    for example in EXAMPLES {
        let text = fs::read_to_string(root.join(example).join(STUDY_FILE_NAME)).unwrap();
        let study: StudyConfig = toml::from_str(&text).unwrap();
        let issues: Vec<String> = validate(&study).iter().map(Issue::to_string).collect();
        assert!(issues.is_empty(), "{example}: {issues:?}");

        for run in expand(&study) {
            let name = format!("{example}/{}", run.dir.display());
            let text = toml::to_string(&run.config).unwrap();
            let mut config: ExperimentConfig =
                toml::from_str(&text).unwrap_or_else(|err| panic!("{name}: {err}"));
            // Tests run in the crate directory, not in the repository root.
            config.data.train = root.join(&config.data.train);
            config.data.test = root.join(&config.data.test);
            let mut issues = crate::config::validate(&config);
            issues.extend(check_files(&config.data).issues);
            let issues: Vec<String> = issues.iter().map(Issue::to_string).collect();
            assert!(issues.is_empty(), "{name}: {issues:?}");
        }
    }
}
