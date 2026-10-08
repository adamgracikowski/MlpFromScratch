//! The data preparation pipeline: load, encode, split, normalize.

use ndarray::Array2;

use super::{
    ClassEncoding, DataError, Dataset, PreparedData, Scaler, Table, read_table, split_indices,
};
use crate::config::{Issue, LoadedConfig, Task};
use crate::rng::{RngStream, rng};

/// Loads the datasets of an experiment and prepares them for training.
///
/// 1. Reads the training and test CSV files.
/// 2. Classification: maps labels to class indices (classes come from the
///    whole training file) and builds one-hot targets.
/// 3. Draws `train_subset` samples, then holds out `validation_fraction` of
///    them for validation; both stratified by class for classification and
///    driven by the [`RngStream::Data`] stream of the seed.
/// 4. Fits normalization on the remaining training samples only and applies
///    it to all three sets.
///
/// A subset or validation split that does not fit the number of samples is
/// reported at its line in `config.toml`.
pub fn prepare(loaded: &LoadedConfig) -> Result<PreparedData, DataError> {
    let data = &loaded.config.data;
    let train_table = read_table(&data.train, &data.target)?;
    let test_table = read_table(&data.test, &data.target)?;
    let feature_names = train_table.feature_names.clone();

    let (classes, full_train, mut test) = match loaded.config.task {
        Task::Classification => {
            let classes = ClassEncoding::fit(&train_table)?;
            let train = classification_dataset(train_table, &classes)?;
            let test = classification_dataset(test_table, &classes)?;
            (Some(classes), train, test)
        }
        Task::Regression => (
            None,
            regression_dataset(train_table),
            regression_dataset(test_table),
        ),
    };

    let (mut train, mut validation) = split(loaded, full_train)?;
    let input_scaler = Scaler::fit(data.input_normalization, &train.inputs);
    let target_scaler = classes
        .is_none()
        .then(|| Scaler::fit(data.target_normalization, &train.targets));
    for dataset in [Some(&mut train), validation.as_mut(), Some(&mut test)]
        .into_iter()
        .flatten()
    {
        dataset.normalize(&input_scaler, target_scaler.as_ref());
    }

    Ok(PreparedData {
        train,
        validation,
        test,
        feature_names,
        input_scaler,
        target_scaler,
        classes,
    })
}

/// Inputs, one-hot targets and class indices of a classification table.
fn classification_dataset(table: Table, classes: &ClassEncoding) -> Result<Dataset, DataError> {
    let labels = classes.encode(&table)?;
    Ok(Dataset {
        inputs: table.features,
        targets: classes.one_hot(&labels),
        labels: Some(labels),
    })
}

/// Inputs and the target column (`n x 1`) of a regression table.
fn regression_dataset(table: Table) -> Dataset {
    let rows = table.len();
    Dataset {
        inputs: table.features,
        targets: Array2::from_shape_vec((rows, 1), table.target).expect("one target per sample"),
        labels: None,
    }
}

/// Draws the training subset and splits off the validation set.
fn split(loaded: &LoadedConfig, full: Dataset) -> Result<(Dataset, Option<Dataset>), DataError> {
    let data = &loaded.config.data;
    let mut rng = rng(loaded.config.training.seed, RngStream::Data);
    let labels = full
        .labels
        .as_ref()
        .map(|labels| labels.as_slice().expect("labels are contiguous"));
    let total = full.len();
    let all: Vec<usize> = (0..total).collect();

    let pool = match data.train_subset {
        None => all,
        Some(subset) if subset > total => {
            let issue = Issue::error(
                &["data", "train_subset"],
                format!(
                    "train_subset ({subset}) exceeds the {total} sample(s) of the training set"
                ),
            )
            .with_help(format!(
                "use at most {total}, or remove `train_subset` to train on the whole set"
            ));
            return Err(loaded.invalid(vec![issue]).into());
        }
        Some(subset) => split_indices(&all, subset, labels, &mut rng).chosen,
    };

    let fraction = data.validation_fraction;
    let available = pool.len();
    let count = (available as f64 * fraction).round() as usize;
    if fraction > 0.0 && (count == 0 || count == available) {
        let message = if count == 0 {
            format!(
                "validation set would be empty: {fraction} of {available} training sample(s) rounds to 0"
            )
        } else {
            format!("no training samples left: validation takes all {available} sample(s)")
        };
        let issue = Issue::error(&["data", "validation_fraction"], message)
            .with_help("change `validation_fraction` or use more training samples");
        return Err(loaded.invalid(vec![issue]).into());
    }

    if count == 0 {
        let train = if available == total {
            full
        } else {
            full.select(&pool)
        };
        return Ok((train, None));
    }
    let parts = split_indices(&pool, count, labels, &mut rng);
    Ok((full.select(&parts.rest), Some(full.select(&parts.chosen))))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use ndarray::{Array1, Axis, array};
    use tempfile::TempDir;

    use super::*;
    use crate::config::{CLASSIFICATION, ConfigError, load_experiment};

    const TEST: &str = "x,y,cls\n0,0,1\n15,30,2\n30,60,3\n";

    /// Prepares data from `config` with 30 training samples, 10 per class
    /// (labels 1, 2, 3 interleaved, features `x = i` and `y = 2i + label`),
    /// and the test file contents `test`.
    fn prepare_files(config: &str, test: &str) -> Result<PreparedData, DataError> {
        let dir = TempDir::new().unwrap();
        let mut train = String::from("x,y,cls\n");
        for i in 0..30 {
            let label = i % 3 + 1;
            train.push_str(&format!("{i},{},{label}\n", 2 * i + label));
        }
        fs::write(dir.path().join("train.csv"), train).unwrap();
        fs::write(dir.path().join("test.csv"), test).unwrap();
        let path = |name: &str| format!("'{}'", dir.path().join(name).display());
        let text = config
            .replace("\"train.csv\"", &path("train.csv"))
            .replace("\"test.csv\"", &path("test.csv"));
        fs::write(dir.path().join("config.toml"), text).unwrap();
        prepare(&load_experiment(dir.path()).unwrap())
    }

    fn prepare_with(config: &str) -> Result<PreparedData, DataError> {
        prepare_files(config, TEST)
    }

    fn classification(extra: &str) -> String {
        CLASSIFICATION.replace("validation_fraction = 0.0", extra)
    }

    fn regression() -> String {
        CLASSIFICATION
            .replace("task = \"classification\"", "task = \"regression\"")
            .replace(
                "target_normalization = \"none\"",
                "target_normalization = \"standard\"",
            )
            .replace("\"softmax\"", "\"identity\"")
            .replace("\"cross_entropy\"", "\"mse\"")
    }

    fn mean(data: &Array2<f64>) -> Array1<f64> {
        data.mean_axis(Axis(0)).unwrap()
    }

    fn assert_issue_at(err: DataError, expected: &str) {
        let DataError::Config(err) = err else {
            panic!("expected a config error, got {err:?}");
        };
        let ConfigError::Invalid { issues, .. } = *err else {
            panic!("expected validation errors, got {err:?}");
        };
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].issue().path_string(), expected);
        assert!(
            issues[0].offset().is_some(),
            "issue is not located in the config"
        );
    }

    #[test]
    fn prepares_classification_data() {
        let prepared = prepare_with(CLASSIFICATION).unwrap();
        let train = &prepared.train;
        assert_eq!(prepared.feature_names, ["x", "y"]);
        assert_eq!(train.inputs.dim(), (30, 2));
        assert_eq!(train.targets.dim(), (30, 3));
        assert_eq!(train.targets.row(4), array![0.0, 1.0, 0.0]);
        assert_eq!(train.class_counts(3), Some(vec![10, 10, 10]));
        assert_eq!(prepared.classes.unwrap().classes(), [1, 2, 3]);
        assert!(prepared.validation.is_none());
        assert!(prepared.target_scaler.is_none());
        assert!(mean(&train.inputs).iter().all(|m| m.abs() < 1e-12));
        assert_eq!(prepared.test.labels, Some(array![0, 1, 2]));
    }

    #[test]
    fn holds_out_stratified_validation() {
        let prepared = prepare_with(&classification("validation_fraction = 0.2")).unwrap();
        let validation = prepared.validation.unwrap();
        assert_eq!(prepared.train.class_counts(3), Some(vec![8, 8, 8]));
        assert_eq!(validation.class_counts(3), Some(vec![2, 2, 2]));
        // Normalization is fitted on the remaining training samples only.
        assert!(mean(&prepared.train.inputs).iter().all(|m| m.abs() < 1e-12));
    }

    #[test]
    fn draws_stratified_subset() {
        let prepared = prepare_with(&classification(
            "validation_fraction = 0.2\ntrain_subset = 15",
        ))
        .unwrap();
        assert_eq!(prepared.train.class_counts(3), Some(vec![4, 4, 4]));
        assert_eq!(
            prepared.validation.unwrap().class_counts(3),
            Some(vec![1, 1, 1])
        );
    }

    #[test]
    fn applies_training_statistics_to_test_set() {
        let prepared = prepare_with(CLASSIFICATION).unwrap();
        let scaler = &prepared.input_scaler;
        let raw_test = array![[0.0, 0.0], [15.0, 30.0], [30.0, 60.0]];
        assert_eq!(prepared.test.inputs, scaler.transform(&raw_test));
        assert_eq!(scaler.offset[0], 14.5);
    }

    #[test]
    fn same_seed_gives_same_data() {
        let config = classification("validation_fraction = 0.2\ntrain_subset = 20");
        assert_eq!(
            prepare_with(&config).unwrap(),
            prepare_with(&config).unwrap()
        );
        let other_seed = config.replace("seed = 42", "seed = 7");
        assert_ne!(
            prepare_with(&config).unwrap().validation,
            prepare_with(&other_seed).unwrap().validation
        );
    }

    #[test]
    fn normalizes_regression_target() {
        let prepared = prepare_with(&regression()).unwrap();
        let train = &prepared.train;
        assert!(train.labels.is_none() && prepared.classes.is_none());
        assert_eq!(train.targets.dim(), (30, 1));
        assert!(mean(&train.targets)[0].abs() < 1e-12);
        let scaler = prepared.target_scaler.unwrap();
        let restored = scaler.inverse_transform(&train.targets);
        let raw: Vec<f64> = (0..30).map(|i| f64::from(i % 3 + 1)).collect();
        for (restored, raw) in restored.iter().zip(&raw) {
            assert!((restored - raw).abs() < 1e-12);
        }
    }

    #[test]
    fn reports_subset_larger_than_training_set() {
        let err = prepare_with(&classification(
            "validation_fraction = 0.0\ntrain_subset = 31",
        ))
        .unwrap_err();
        assert_issue_at(err, "data.train_subset");
    }

    #[test]
    fn reports_empty_validation_set() {
        let err = prepare_with(&classification("validation_fraction = 0.01")).unwrap_err();
        assert_issue_at(err, "data.validation_fraction");
    }

    #[test]
    fn reports_validation_taking_all_samples() {
        let err = prepare_with(&classification(
            "validation_fraction = 0.6\ntrain_subset = 1",
        ))
        .unwrap_err();
        assert_issue_at(err, "data.validation_fraction");
    }

    #[test]
    fn reports_unknown_test_label() {
        let err = prepare_files(CLASSIFICATION, "x,y,cls\n0,0,4\n").unwrap_err();
        assert!(
            matches!(err, DataError::UnknownLabel { label: 4, .. }),
            "{err:?}"
        );
    }
}
