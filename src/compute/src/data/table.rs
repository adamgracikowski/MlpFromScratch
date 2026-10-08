//! Reading CSV files into a feature matrix and a target column.

use std::fs::File;
use std::path::{Path, PathBuf};

use csv::{Reader, ReaderBuilder, StringRecord, Trim};
use ndarray::Array2;

use super::DataError;

/// Contents of a CSV dataset: input features and the target column.
#[derive(Debug, Clone, PartialEq)]
pub struct Table {
    /// File the table was read from, for error messages.
    pub path: PathBuf,
    /// Names of the feature columns, in file order.
    pub feature_names: Vec<String>,
    /// Input features, one row per sample (`n x d`).
    pub features: Array2<f64>,
    /// Target value of every sample (`n`).
    pub target: Vec<f64>,
    /// Line in the file where each sample starts (1-based), for error messages.
    pub lines: Vec<u64>,
}

impl Table {
    /// Number of samples.
    pub fn len(&self) -> usize {
        self.target.len()
    }

    /// Whether the table has no samples.
    pub fn is_empty(&self) -> bool {
        self.target.is_empty()
    }
}

/// Opens a CSV reader with the settings shared by all dataset files: a header
/// row, trimmed fields, optional quotes, CRLF or LF line endings and an
/// optional UTF-8 byte order mark.
fn open(path: &Path) -> csv::Result<Reader<File>> {
    ReaderBuilder::new().trim(Trim::All).from_path(path)
}

/// Reads the column names from the header, or an empty list for an empty file.
pub fn read_header(path: &Path) -> csv::Result<Vec<String>> {
    let mut reader = open(path)?;
    Ok(reader.headers()?.iter().map(str::to_owned).collect())
}

/// Reads a whole CSV file. Column `target` becomes the target, all other
/// columns become features. Every value must be a finite number.
pub fn read_table(path: &Path, target: &str) -> Result<Table, DataError> {
    let read_error = |source| DataError::Read {
        path: path.to_owned(),
        source,
    };
    let mut reader = open(path).map_err(read_error)?;
    let header = reader.headers().map_err(read_error)?.clone();
    if header.is_empty() {
        return Err(DataError::EmptyFile {
            path: path.to_owned(),
        });
    }
    let Some(target_index) = header.iter().position(|column| column == target) else {
        return Err(DataError::MissingColumn {
            path: path.to_owned(),
            column: target.to_owned(),
            available: header.iter().collect::<Vec<_>>().join(", "),
        });
    };
    let feature_names: Vec<String> = header
        .iter()
        .enumerate()
        .filter(|&(index, _)| index != target_index)
        .map(|(_, column)| column.to_owned())
        .collect();
    if feature_names.is_empty() {
        return Err(DataError::NoFeatures {
            path: path.to_owned(),
        });
    }

    let mut features = Vec::new();
    let mut targets = Vec::new();
    let mut lines = Vec::new();
    let mut record = StringRecord::new();
    while reader.read_record(&mut record).map_err(read_error)? {
        let line = record.position().map_or(0, |position| position.line());
        for (index, field) in record.iter().enumerate() {
            let value = parse_value(field).ok_or_else(|| DataError::InvalidValue {
                path: path.to_owned(),
                line,
                column: header[index].to_owned(),
                value: field.to_owned(),
            })?;
            if index == target_index {
                targets.push(value);
            } else {
                features.push(value);
            }
        }
        lines.push(line);
    }
    if targets.is_empty() {
        return Err(DataError::NoRows {
            path: path.to_owned(),
        });
    }

    let features = Array2::from_shape_vec((targets.len(), feature_names.len()), features)
        .expect("every record has as many fields as the header");
    Ok(Table {
        path: path.to_owned(),
        feature_names,
        features,
        target: targets,
        lines,
    })
}

/// Parses a finite floating-point number.
fn parse_value(field: &str) -> Option<f64> {
    field.parse::<f64>().ok().filter(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use ndarray::array;
    use tempfile::TempDir;

    use super::*;

    fn write(dir: &TempDir, contents: &str) -> PathBuf {
        let path = dir.path().join("data.csv");
        fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn reads_features_and_target() {
        let dir = TempDir::new().unwrap();
        let path = write(&dir, "x,cls,y\n0.5,1,-2\n1e-3,2,3.25\n");
        let table = read_table(&path, "cls").unwrap();
        assert_eq!(table.feature_names, ["x", "y"]);
        assert_eq!(table.features, array![[0.5, -2.0], [0.001, 3.25]]);
        assert_eq!(table.target, [1.0, 2.0]);
        assert_eq!(table.lines, [2, 3]);
        assert_eq!(table.len(), 2);
    }

    #[test]
    fn handles_bom_quotes_crlf_and_spaces() {
        let dir = TempDir::new().unwrap();
        let path = write(&dir, "\u{feff}\"x\",\"y\"\r\n\"0.1\", 7 \r\n");
        let table = read_table(&path, "y").unwrap();
        assert_eq!(table.feature_names, ["x"]);
        assert_eq!(table.features, array![[0.1]]);
        assert_eq!(table.target, [7.0]);
    }

    #[test]
    fn reports_invalid_value_with_line_and_column() {
        let dir = TempDir::new().unwrap();
        let path = write(&dir, "x,y\n1,2\n3,abc\n");
        let err = read_table(&path, "y").unwrap_err();
        let DataError::InvalidValue {
            line,
            column,
            value,
            ..
        } = err
        else {
            panic!("expected an invalid value, got {err:?}");
        };
        assert_eq!((line, column.as_str(), value.as_str()), (3, "y", "abc"));
    }

    #[test]
    fn rejects_non_finite_and_empty_values() {
        let dir = TempDir::new().unwrap();
        for row in ["NaN,1", "inf,1", ",1"] {
            let path = write(&dir, &format!("x,y\n{row}\n"));
            let err = read_table(&path, "y").unwrap_err();
            assert!(
                matches!(err, DataError::InvalidValue { .. }),
                "{row}: {err:?}"
            );
        }
    }

    #[test]
    fn reports_wrong_field_count() {
        let dir = TempDir::new().unwrap();
        let path = write(&dir, "x,y\n1,2\n3\n");
        let err = read_table(&path, "y").unwrap_err();
        assert!(matches!(err, DataError::Read { .. }), "{err:?}");
    }

    #[test]
    fn reports_missing_file() {
        let dir = TempDir::new().unwrap();
        let err = read_table(&dir.path().join("missing.csv"), "y").unwrap_err();
        assert!(matches!(err, DataError::Read { .. }), "{err:?}");
    }

    #[test]
    fn reports_empty_file_and_missing_rows() {
        let dir = TempDir::new().unwrap();
        let err = read_table(&write(&dir, ""), "y").unwrap_err();
        assert!(matches!(err, DataError::EmptyFile { .. }), "{err:?}");
        let err = read_table(&write(&dir, "x,y\n"), "y").unwrap_err();
        assert!(matches!(err, DataError::NoRows { .. }), "{err:?}");
    }

    #[test]
    fn reports_missing_target_and_features() {
        let dir = TempDir::new().unwrap();
        let err = read_table(&write(&dir, "x,y\n1,2\n"), "cls").unwrap_err();
        let DataError::MissingColumn { available, .. } = err else {
            panic!("expected a missing column, got {err:?}");
        };
        assert_eq!(available, "x, y");
        let err = read_table(&write(&dir, "y\n2\n"), "y").unwrap_err();
        assert!(matches!(err, DataError::NoFeatures { .. }), "{err:?}");
    }

    #[test]
    fn reads_header_only() {
        let dir = TempDir::new().unwrap();
        let path = write(&dir, "\u{feff}a ,\"b\"\r\nnot,checked\n");
        assert_eq!(read_header(&path).unwrap(), ["a", "b"]);
        assert!(read_header(&write(&dir, "")).unwrap().is_empty());
    }
}
