//! Mapping of integer class labels to indices and one-hot vectors.

use ndarray::{Array1, Array2};

use super::{DataError, Table};

/// Minimum number of classes in a classification dataset.
pub const MIN_CLASSES: usize = 2;

/// Class labels found in the training set, sorted ascending.
///
/// Class `i` of the network output corresponds to label `classes[i]`, so
/// 1-based labels (`1, 2, 3`) and 0-based labels (MNIST `0..=9`) both map to
/// indices `0..K`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassEncoding {
    classes: Vec<i64>,
}

impl ClassEncoding {
    /// Collects the distinct labels of the target column of `table`.
    pub fn fit(table: &Table) -> Result<Self, DataError> {
        let mut classes = (0..table.len())
            .map(|row| label(table, row))
            .collect::<Result<Vec<_>, _>>()?;
        classes.sort_unstable();
        classes.dedup();
        if classes.len() < MIN_CLASSES {
            return Err(DataError::TooFewClasses {
                path: table.path.clone(),
                classes: join(&classes),
            });
        }
        Ok(Self { classes })
    }

    /// Sorted class labels.
    pub fn classes(&self) -> &[i64] {
        &self.classes
    }

    /// Number of classes `K`.
    pub fn class_count(&self) -> usize {
        self.classes.len()
    }

    /// Converts the target column of `table` to class indices `0..K`.
    /// A label not seen in the training set is an error.
    pub fn encode(&self, table: &Table) -> Result<Array1<usize>, DataError> {
        (0..table.len())
            .map(|row| {
                let label = label(table, row)?;
                self.classes
                    .binary_search(&label)
                    .map_err(|_| DataError::UnknownLabel {
                        path: table.path.clone(),
                        line: table.lines[row],
                        label,
                        classes: join(&self.classes),
                    })
            })
            .collect()
    }

    /// One-hot matrix (`n × K`) of class indices: row `i` has 1 in column
    /// `labels[i]` and 0 elsewhere.
    pub fn one_hot(&self, labels: &Array1<usize>) -> Array2<f64> {
        let mut one_hot = Array2::zeros((labels.len(), self.class_count()));
        for (row, &label) in labels.iter().enumerate() {
            one_hot[[row, label]] = 1.0;
        }
        one_hot
    }
}

/// Reads the target of sample `row` as an integer label.
fn label(table: &Table, row: usize) -> Result<i64, DataError> {
    /// Largest magnitude below which every integer is exact in `f64`.
    const MAX_EXACT: f64 = (1u64 << 53) as f64;
    let value = table.target[row];
    if value.fract() != 0.0 || value.abs() > MAX_EXACT {
        return Err(DataError::NonIntegerLabel {
            path: table.path.clone(),
            line: table.lines[row],
            value,
        });
    }
    Ok(value as i64)
}

fn join(classes: &[i64]) -> String {
    classes
        .iter()
        .map(i64::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use ndarray::array;

    use super::*;

    fn table(target: &[f64]) -> Table {
        Table {
            path: PathBuf::from("data.csv"),
            feature_names: vec!["x".to_owned()],
            features: Array2::zeros((target.len(), 1)),
            target: target.to_vec(),
            lines: (2..).take(target.len()).collect(),
        }
    }

    #[test]
    fn maps_one_based_labels_to_indices() {
        let train = table(&[2.0, 1.0, 3.0, 1.0]);
        let encoding = ClassEncoding::fit(&train).unwrap();
        assert_eq!(encoding.classes(), [1, 2, 3]);
        assert_eq!(encoding.class_count(), 3);
        assert_eq!(encoding.encode(&train).unwrap(), array![1, 0, 2, 0]);
    }

    #[test]
    fn maps_zero_based_labels_to_indices() {
        let train = table(&[9.0, 0.0, 5.0]);
        let encoding = ClassEncoding::fit(&train).unwrap();
        assert_eq!(encoding.classes(), [0, 5, 9]);
        assert_eq!(encoding.encode(&train).unwrap(), array![2, 0, 1]);
    }

    #[test]
    fn builds_one_hot_matrix() {
        let encoding = ClassEncoding::fit(&table(&[1.0, 2.0, 3.0])).unwrap();
        let one_hot = encoding.one_hot(&array![2, 0]);
        assert_eq!(one_hot, array![[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]);
    }

    #[test]
    fn rejects_non_integer_label() {
        let err = ClassEncoding::fit(&table(&[1.0, 1.5])).unwrap_err();
        let DataError::NonIntegerLabel { line, value, .. } = err else {
            panic!("expected a non-integer label, got {err:?}");
        };
        assert_eq!((line, value), (3, 1.5));
    }

    #[test]
    fn rejects_single_class() {
        let err = ClassEncoding::fit(&table(&[1.0, 1.0])).unwrap_err();
        assert!(matches!(err, DataError::TooFewClasses { .. }), "{err:?}");
    }

    #[test]
    fn rejects_label_unknown_in_training_set() {
        let encoding = ClassEncoding::fit(&table(&[1.0, 2.0])).unwrap();
        let err = encoding.encode(&table(&[2.0, 3.0])).unwrap_err();
        let DataError::UnknownLabel {
            line,
            label,
            classes,
            ..
        } = err
        else {
            panic!("expected an unknown label, got {err:?}");
        };
        assert_eq!((line, label, classes.as_str()), (3, 3, "1, 2"));
    }
}
