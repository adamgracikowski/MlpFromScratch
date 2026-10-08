//! Checks of the dataset files referenced by a config.
//!
//! Only the CSV header is read, so the check stays fast for large datasets
//! such as MNIST. Loading the data itself is left to the `data` module.

use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

use super::{DataConfig, Issue};

/// Result of checking the dataset files.
#[derive(Debug, Default)]
pub struct FileCheck {
    /// Problems found, reported at `data.train`, `data.test` or `data.target`.
    pub issues: Vec<Issue>,
    /// Number of input feature columns, known once the training header is valid.
    pub feature_count: Option<usize>,
}

/// Checks that both CSV files exist, contain the target column and at least
/// one feature column, and share the same columns.
pub fn check_files(data: &DataConfig) -> FileCheck {
    let mut check = FileCheck::default();
    let train = check_file(data, "train", &data.train, &mut check.issues);
    let test = check_file(data, "test", &data.test, &mut check.issues);

    if let (Some(train), Some(test)) = (&train, &test) {
        if train != test {
            check.issues.push(
                Issue::error(
                    &["data", "test"],
                    format!(
                        "columns ({}) differ from the training set ({})",
                        test.join(", "),
                        train.join(", ")
                    ),
                )
                .with_help("both files must have the same columns in the same order"),
            );
        }
    }
    check.feature_count = train.map(|columns| columns.len() - 1);
    check
}

/// Checks one CSV file and returns its columns when they are usable.
fn check_file(
    data: &DataConfig,
    field: &'static str,
    path: &Path,
    issues: &mut Vec<Issue>,
) -> Option<Vec<String>> {
    let columns = match read_header(path) {
        Ok(Some(columns)) => columns,
        Ok(None) => {
            issues.push(Issue::error(
                &["data", field],
                format!("`{}` is empty", path.display()),
            ));
            return None;
        }
        Err(err) => {
            issues.push(
                Issue::error(
                    &["data", field],
                    format!("cannot open `{}`: {err}", path.display()),
                )
                .with_help("paths are resolved relative to the current working directory"),
            );
            return None;
        }
    };

    if !columns.contains(&data.target) {
        issues.push(
            Issue::error(
                &["data", "target"],
                format!("column `{}` not found in `{}`", data.target, path.display()),
            )
            .with_help(format!("available columns: {}", columns.join(", "))),
        );
        return None;
    }
    if columns.len() < 2 {
        issues.push(Issue::error(
            &["data", field],
            format!("`{}` has no feature columns", path.display()),
        ));
        return None;
    }
    Some(columns)
}

/// Reads the column names from the first line, or `None` for an empty file.
fn read_header(path: &Path) -> io::Result<Option<Vec<String>>> {
    let mut line = String::new();
    if BufReader::new(File::open(path)?).read_line(&mut line)? == 0 {
        return Ok(None);
    }
    let line = line.trim_start_matches('\u{feff}').trim_end();
    let columns = line
        .split(',')
        .map(|column| column.trim().trim_matches('"').to_owned())
        .collect();
    Ok(Some(columns))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use tempfile::TempDir;

    use super::*;
    use crate::config::Normalization;

    fn write(dir: &TempDir, name: &str, contents: &str) -> PathBuf {
        let path = dir.path().join(name);
        fs::write(&path, contents).unwrap();
        path
    }

    fn data(train: PathBuf, test: PathBuf) -> DataConfig {
        DataConfig {
            train,
            test,
            target: "cls".to_owned(),
            input_normalization: Normalization::None,
            target_normalization: Normalization::None,
            train_subset: None,
            validation_fraction: 0.0,
        }
    }

    fn paths(check: &FileCheck) -> Vec<String> {
        check.issues.iter().map(Issue::path_string).collect()
    }

    #[test]
    fn accepts_matching_headers() {
        let dir = TempDir::new().unwrap();
        let train = write(&dir, "train.csv", "x,y,cls\n0.1,0.2,1\n");
        let test = write(
            &dir,
            "test.csv",
            "\u{feff}\"x\",\"y\",\"cls\"\r\n0.3,0.4,2\r\n",
        );
        let check = check_files(&data(train, test));
        assert!(check.issues.is_empty(), "{:?}", check.issues);
        assert_eq!(check.feature_count, Some(2));
    }

    #[test]
    fn reports_missing_file() {
        let dir = TempDir::new().unwrap();
        let train = write(&dir, "train.csv", "x,y,cls\n");
        let check = check_files(&data(train, dir.path().join("missing.csv")));
        assert_eq!(paths(&check), ["data.test"]);
        assert_eq!(check.feature_count, Some(2));
    }

    #[test]
    fn reports_empty_file() {
        let dir = TempDir::new().unwrap();
        let train = write(&dir, "train.csv", "");
        let test = write(&dir, "test.csv", "x,y,cls\n");
        let check = check_files(&data(train, test));
        assert_eq!(paths(&check), ["data.train"]);
        assert_eq!(check.feature_count, None);
    }

    #[test]
    fn reports_missing_target_column() {
        let dir = TempDir::new().unwrap();
        let train = write(&dir, "train.csv", "x,y\n");
        let test = write(&dir, "test.csv", "x,y\n");
        let check = check_files(&data(train, test));
        assert_eq!(paths(&check), ["data.target", "data.target"]);
        assert!(check.issues[0].help.as_deref().unwrap().contains("x, y"));
    }

    #[test]
    fn reports_missing_feature_columns() {
        let dir = TempDir::new().unwrap();
        let train = write(&dir, "train.csv", "cls\n1\n");
        let test = write(&dir, "test.csv", "x,cls\n");
        let check = check_files(&data(train, test));
        assert_eq!(paths(&check), ["data.train"]);
    }

    #[test]
    fn reports_different_columns() {
        let dir = TempDir::new().unwrap();
        let train = write(&dir, "train.csv", "x,y,cls\n");
        let test = write(&dir, "test.csv", "y,x,cls\n");
        let check = check_files(&data(train, test));
        assert_eq!(paths(&check), ["data.test"]);
    }
}
