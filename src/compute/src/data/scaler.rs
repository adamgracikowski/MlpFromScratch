//! Per-column normalization of features and targets.

use ndarray::{Array1, Array2, Axis};

use crate::config::Normalization;

/// Affine per-column transformation `(x - offset) / scale`.
///
/// Statistics are fitted on the training set only and then applied unchanged
/// to the validation and test sets. A constant column keeps scale 1, so it is
/// only shifted (to 0) instead of being divided by zero.
#[derive(Debug, Clone, PartialEq)]
pub struct Scaler {
    /// Kind of normalization the statistics were fitted for.
    pub normalization: Normalization,
    /// Value subtracted from each column (mean or minimum).
    pub offset: Array1<f64>,
    /// Divisor of each column (standard deviation or range).
    pub scale: Array1<f64>,
}

impl Scaler {
    /// Fits per-column statistics of `data` (`n x d`, `n >= 1`).
    pub fn fit(normalization: Normalization, data: &Array2<f64>) -> Self {
        assert!(data.nrows() > 0, "cannot fit a scaler on an empty matrix");
        let columns = data.ncols();
        let (offset, scale) = match normalization {
            Normalization::None => (Array1::zeros(columns), Array1::ones(columns)),
            Normalization::Standard => (
                data.mean_axis(Axis(0)).expect("data is not empty"),
                data.std_axis(Axis(0), 0.0),
            ),
            Normalization::MinMax => {
                let min = column_fold(data, f64::INFINITY, f64::min);
                let max = column_fold(data, f64::NEG_INFINITY, f64::max);
                let range = &max - &min;
                (min, range)
            }
        };
        let constant = constant_columns(data);
        let scale = scale
            .iter()
            .zip(&constant)
            .map(|(&scale, &constant)| if constant { 1.0 } else { scale })
            .collect();
        Self {
            normalization,
            offset,
            scale,
        }
    }

    /// Normalizes `data`: `(x - offset) / scale` in every column.
    pub fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let mut data = data.clone();
        self.transform_in_place(&mut data);
        data
    }

    /// Like [`Scaler::transform`], but overwrites `data` instead of
    /// allocating a copy (large datasets such as MNIST).
    pub fn transform_in_place(&self, data: &mut Array2<f64>) {
        self.assert_columns(data);
        *data -= &self.offset;
        *data /= &self.scale;
    }

    /// Reverts [`Scaler::transform`], e.g. to bring predictions back to the
    /// original scale of the target.
    pub fn inverse_transform(&self, data: &Array2<f64>) -> Array2<f64> {
        self.assert_columns(data);
        data * &self.scale + &self.offset
    }

    fn assert_columns(&self, data: &Array2<f64>) {
        assert_eq!(
            data.ncols(),
            self.offset.len(),
            "scaler fitted for {} column(s), got {}",
            self.offset.len(),
            data.ncols()
        );
    }
}

/// Folds every column of `data` into one value.
fn column_fold(data: &Array2<f64>, init: f64, fold: fn(f64, f64) -> f64) -> Array1<f64> {
    data.fold_axis(Axis(0), init, |&acc, &value| fold(acc, value))
}

/// Marks columns whose values are all equal.
fn constant_columns(data: &Array2<f64>) -> Vec<bool> {
    data.columns()
        .into_iter()
        .map(|column| column.iter().all(|&value| value == column[0]))
        .collect()
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    const TOLERANCE: f64 = 1e-12;

    fn assert_close(actual: &Array2<f64>, expected: &Array2<f64>) {
        assert_eq!(actual.dim(), expected.dim());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < TOLERANCE, "{actual} != {expected}");
        }
    }

    fn data() -> Array2<f64> {
        array![[1.0, 10.0, 5.0], [2.0, 30.0, 5.0], [3.0, 20.0, 5.0]]
    }

    #[test]
    fn none_leaves_values_unchanged() {
        let scaler = Scaler::fit(Normalization::None, &data());
        assert_close(&scaler.transform(&data()), &data());
    }

    #[test]
    fn standard_gives_zero_mean_and_unit_variance() {
        let scaler = Scaler::fit(Normalization::Standard, &data());
        let scaled = scaler.transform(&data());
        let mean = scaled.mean_axis(Axis(0)).unwrap();
        let std = scaled.std_axis(Axis(0), 0.0);
        assert!(mean.iter().all(|m| m.abs() < TOLERANCE), "{mean}");
        assert!((std[0] - 1.0).abs() < TOLERANCE && (std[1] - 1.0).abs() < TOLERANCE);
    }

    #[test]
    fn min_max_maps_to_unit_interval() {
        let scaler = Scaler::fit(Normalization::MinMax, &data());
        let expected = array![[0.0, 0.0, 0.0], [0.5, 1.0, 0.0], [1.0, 0.5, 0.0]];
        assert_close(&scaler.transform(&data()), &expected);
    }

    #[test]
    fn constant_column_is_shifted_to_zero() {
        for normalization in [Normalization::Standard, Normalization::MinMax] {
            let scaler = Scaler::fit(normalization, &data());
            assert_eq!(scaler.scale[2], 1.0);
            assert!(
                scaler
                    .transform(&data())
                    .column(2)
                    .iter()
                    .all(|&v| v == 0.0)
            );
        }
    }

    #[test]
    fn inverse_transform_reverts_transform() {
        for normalization in [
            Normalization::None,
            Normalization::Standard,
            Normalization::MinMax,
        ] {
            let scaler = Scaler::fit(normalization, &data());
            assert_close(
                &scaler.inverse_transform(&scaler.transform(&data())),
                &data(),
            );
        }
    }

    #[test]
    fn applies_training_statistics_to_other_data() {
        let scaler = Scaler::fit(Normalization::MinMax, &array![[0.0], [10.0]]);
        assert_close(
            &scaler.transform(&array![[20.0], [-5.0]]),
            &array![[2.0], [-0.5]],
        );
    }

    #[test]
    #[should_panic(expected = "scaler fitted for 3 column(s), got 1")]
    fn rejects_wrong_column_count() {
        Scaler::fit(Normalization::Standard, &data()).transform(&array![[1.0]]);
    }
}
