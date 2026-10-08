//! Checks that the example experiment configs in `data/experiments/` stay valid.

use std::path::{Path, PathBuf};

use super::{ExperimentConfig, Issue, check_files, validate};

/// Example configs, relative to the repository root.
const EXAMPLES: [&str; 2] = [
    "data/experiments/classification/example-three-gauss/config.toml",
    "data/experiments/regression/example-cube/config.toml",
];

/// The repository root; dataset paths in configs are relative to it.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn example_configs_are_valid() {
    let root = repo_root();
    for example in EXAMPLES {
        let text = std::fs::read_to_string(root.join(example)).unwrap();
        let mut config: ExperimentConfig = toml::from_str(&text).unwrap();
        // Tests run in the crate directory, not in the repository root.
        config.data.train = root.join(&config.data.train);
        config.data.test = root.join(&config.data.test);

        let mut issues = validate(&config);
        issues.extend(check_files(&config.data).issues);
        let issues: Vec<String> = issues.iter().map(Issue::to_string).collect();
        assert!(issues.is_empty(), "{example}: {issues:?}");
    }
}
