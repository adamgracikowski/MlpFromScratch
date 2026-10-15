//! Loading and checking the config of an experiment directory.

use std::fs;
use std::path::{Path, PathBuf};

use miette::NamedSource;
use toml::de::DeTable;

use super::{
    CONFIG_FILE_NAME, ConfigError, ExperimentConfig, Issue, IssueDiagnostic, check_files, validate,
};

/// A config that passed validation, with the warnings it produced.
#[derive(Debug)]
pub struct LoadedConfig {
    /// Path of the loaded `config.toml`.
    pub path: PathBuf,
    /// The parsed config.
    pub config: ExperimentConfig,
    /// Non-fatal issues (warnings and advice), in file order.
    pub warnings: Vec<IssueDiagnostic>,
    /// Number of input feature columns in the datasets.
    pub feature_count: usize,
    /// Text of `config.toml`, for diagnostics found after loading.
    pub source: NamedSource<String>,
}

impl LoadedConfig {
    /// Builds an error from issues found after loading, e.g. ones that need
    /// the number of samples in the datasets. Like load errors, every issue
    /// points at its line in `config.toml`.
    pub fn invalid(&self, issues: Vec<Issue>) -> ConfigError {
        let issues = diagnostics(issues, &self.source);
        ConfigError::Invalid {
            path: self.path.clone(),
            error_count: issues.iter().filter(|d| d.issue().is_error()).count(),
            issues,
        }
    }
}

/// Reads `config.toml` from `experiment_dir`, parses it, validates it and
/// checks the headers of the dataset files it references.
///
/// All problems are reported together: the error lists every invalid value,
/// each pointing at its line in `config.toml`.
pub fn load_experiment(experiment_dir: &Path) -> Result<LoadedConfig, ConfigError> {
    let path = experiment_dir.join(CONFIG_FILE_NAME);
    let text = fs::read_to_string(&path).map_err(|source| ConfigError::Io {
        path: path.clone(),
        source,
    })?;
    load_from_str(path, text)
}

/// Parses and checks config `text` read from `path`.
pub fn load_from_str(path: PathBuf, text: String) -> Result<LoadedConfig, ConfigError> {
    let src = NamedSource::new(path.display().to_string(), text.clone());
    let config: ExperimentConfig = toml::from_str(&text).map_err(|err| ConfigError::Parse {
        message: err.message().to_owned(),
        src: src.clone(),
        span: err.span().map(Into::into),
    })?;

    let mut issues = validate(&config);
    let files = check_files(&config.data);
    issues.extend(files.issues);

    let diagnostics = diagnostics(issues, &src);

    let error_count = diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.issue().is_error())
        .count();
    if error_count > 0 {
        return Err(ConfigError::Invalid {
            path,
            error_count,
            issues: diagnostics,
        });
    }

    Ok(LoadedConfig {
        path,
        config,
        warnings: diagnostics,
        feature_count: files
            .feature_count
            .expect("a config without errors has a valid training header"),
        source: src,
    })
}

/// Locates `issues` in the TOML text `src` and sorts them by position.
pub(crate) fn diagnostics(issues: Vec<Issue>, src: &NamedSource<String>) -> Vec<IssueDiagnostic> {
    let document = DeTable::parse(src.inner()).ok();
    let mut diagnostics: Vec<IssueDiagnostic> = issues
        .into_iter()
        .map(|issue| IssueDiagnostic::new(issue, document.as_ref(), src.clone()))
        .collect();
    diagnostics.sort_by_key(|diagnostic| diagnostic.offset().unwrap_or(usize::MAX));
    diagnostics
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::config::experiment::tests::CLASSIFICATION;

    /// An experiment directory whose config points at temporary CSV files.
    fn experiment(config: &str) -> TempDir {
        let dir = TempDir::new().unwrap();
        let train = dir.path().join("train.csv");
        let test = dir.path().join("test.csv");
        fs::write(&train, "x,y,cls\n0.1,0.2,1\n").unwrap();
        fs::write(&test, "x,y,cls\n0.3,0.4,2\n").unwrap();
        let text = config
            .replace("\"train.csv\"", &format!("'{}'", train.display()))
            .replace("\"test.csv\"", &format!("'{}'", test.display()));
        fs::write(dir.path().join(CONFIG_FILE_NAME), text).unwrap();
        dir
    }

    #[test]
    fn loads_valid_experiment() {
        let dir = experiment(CLASSIFICATION);
        let loaded = load_experiment(dir.path()).unwrap();
        assert_eq!(loaded.feature_count, 2);
        assert!(loaded.warnings.is_empty());
        assert_eq!(loaded.path, dir.path().join(CONFIG_FILE_NAME));
    }

    #[test]
    fn keeps_warnings_of_valid_experiment() {
        let dir = experiment(&CLASSIFICATION.replace("{ kind = \"he\" }", "{ kind = \"xavier\" }"));
        let loaded = load_experiment(dir.path()).unwrap();
        assert_eq!(loaded.warnings.len(), 1);
        assert_eq!(loaded.warnings[0].issue().path_string(), "model.init");
    }

    #[test]
    fn reports_later_issues_at_config_lines() {
        let dir = experiment(CLASSIFICATION);
        let loaded = load_experiment(dir.path()).unwrap();
        let err = loaded.invalid(vec![
            Issue::warning(&["training", "seed"], "late warning"),
            Issue::error(&["data", "validation_fraction"], "late error"),
        ]);
        let ConfigError::Invalid {
            error_count,
            issues,
            ..
        } = err
        else {
            panic!("expected validation errors, got {err:?}");
        };
        assert_eq!(error_count, 1);
        let paths: Vec<String> = issues.iter().map(|d| d.issue().path_string()).collect();
        assert_eq!(paths, ["data.validation_fraction", "training.seed"]);
        assert!(issues.iter().all(|d| d.offset().is_some()));
    }

    #[test]
    fn reports_missing_config() {
        let dir = TempDir::new().unwrap();
        let err = load_experiment(dir.path()).unwrap_err();
        assert!(matches!(err, ConfigError::Io { .. }));
    }

    #[test]
    fn reports_parse_error_with_span() {
        let dir = experiment(&CLASSIFICATION.replace("epochs = 200", "epochs = \"many\""));
        let err = load_experiment(dir.path()).unwrap_err();
        let ConfigError::Parse { span, .. } = err else {
            panic!("expected a parse error, got {err:?}");
        };
        assert!(span.is_some());
    }

    #[test]
    fn reports_all_invalid_values_in_file_order() {
        let text = CLASSIFICATION
            .replace("seed = 42", "seed = 42\n")
            .replace("grid_resolution = 200", "grid_resolution = 1")
            .replace("\"softmax\"", "\"sigmoid\"")
            .replace("\"train.csv\"", "\"missing.csv\"");
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join(CONFIG_FILE_NAME), text).unwrap();

        let err = load_experiment(dir.path()).unwrap_err();
        let ConfigError::Invalid {
            error_count,
            issues,
            ..
        } = err
        else {
            panic!("expected validation errors, got {err:?}");
        };
        let paths: Vec<String> = issues
            .iter()
            .map(|diagnostic| diagnostic.issue().path_string())
            .collect();
        assert_eq!(error_count, 4);
        assert_eq!(
            paths,
            [
                "data.train",
                "data.test",
                "model.output_activation",
                "output.grid_resolution"
            ]
        );
        assert!(
            issues
                .iter()
                .all(|diagnostic| diagnostic.offset().is_some())
        );
    }
}
