//! Checks that the example experiment configs in `data/experiments/` stay valid
//! and that their datasets can be prepared.

use std::path::{Path, PathBuf};

use miette::NamedSource;

use super::{ExperimentConfig, Issue, LoadedConfig, check_files, validate};
use crate::data::prepare;

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
        let path = root.join(example);
        let text = std::fs::read_to_string(&path).unwrap();
        let mut config: ExperimentConfig = toml::from_str(&text).unwrap();
        // Tests run in the crate directory, not in the repository root.
        config.data.train = root.join(&config.data.train);
        config.data.test = root.join(&config.data.test);

        let mut issues = validate(&config);
        let files = check_files(&config.data);
        issues.extend(files.issues);
        let issues: Vec<String> = issues.iter().map(Issue::to_string).collect();
        assert!(issues.is_empty(), "{example}: {issues:?}");

        let loaded = LoadedConfig {
            source: NamedSource::new(example, text),
            path,
            config,
            warnings: Vec::new(),
            feature_count: files.feature_count.unwrap(),
        };
        if let Err(err) = prepare(&loaded) {
            panic!("{example}: {err:?}");
        }
    }
}
