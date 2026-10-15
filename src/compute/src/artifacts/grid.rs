//! Predictions on a regular grid over the input space (1 or 2 features):
//! the regression curve or the decision regions.

use std::path::Path;

use ndarray::{Array1, Array2, ArrayD, IxDyn};

use super::npy::write_npy;
use super::{ArtifactError, create_dir, predict_in_chunks};
use crate::config::OutputConfig;
use crate::data::{Dataset, PreparedData};
use crate::network::Mlp;

/// Network output on a grid of points in the original input scale.
#[derive(Debug, Clone, PartialEq)]
pub struct Grid {
    /// Coordinates along every feature: `[x]` or `[x, y]`, each of length
    /// `grid_resolution`.
    pub axes: Vec<Array1<f64>>,
    /// Network output in the original target scale: `res x K` for one
    /// feature, `res x res x K` for two, where `[i, j]` is the point
    /// `(x[j], y[i])` (rows follow `y`, as in an image).
    pub outputs: ArrayD<f64>,
}

impl Grid {
    /// Evaluates `network` on a grid spanning all samples (training,
    /// validation and test) widened by `grid_margin` of the range on every
    /// side. Returns `None` for more than two features, which cannot be
    /// plotted this way.
    pub fn compute(network: &Mlp, data: &PreparedData, output: &OutputConfig) -> Option<Self> {
        let features = data.feature_names.len();
        if !(1..=2).contains(&features) {
            return None;
        }
        let axes: Vec<Array1<f64>> = data_ranges(data)
            .into_iter()
            .map(|(min, max)| {
                let (low, high) = widen(min, max, output.grid_margin);
                Array1::linspace(low, high, output.grid_resolution)
            })
            .collect();

        let mut points = grid_points(&axes);
        data.input_scaler.transform_in_place(&mut points);
        let mut outputs = predict_in_chunks(network, points.view());
        if let Some(scaler) = &data.target_scaler {
            outputs = scaler.inverse_transform(&outputs);
        }
        let mut shape: Vec<usize> = axes.iter().map(Array1::len).collect();
        shape.push(outputs.ncols());
        let outputs = outputs
            .into_shape_with_order(IxDyn(&shape))
            .expect("one output row per grid point");
        Some(Self { axes, outputs })
    }

    /// Writes `x.npy` (and `y.npy` for two features) and `outputs.npy` to `dir`.
    pub(super) fn write(&self, dir: &Path) -> Result<(), ArtifactError> {
        create_dir(dir)?;
        for (axis, name) in self.axes.iter().zip(["x.npy", "y.npy"]) {
            write_npy(&dir.join(name), &[axis.len()], axis.iter().copied())?;
        }
        let shape = self.outputs.shape();
        write_npy(
            &dir.join("outputs.npy"),
            shape,
            self.outputs.iter().copied(),
        )
    }
}

/// Minimum and maximum of every feature over all samples, in the original
/// scale.
fn data_ranges(data: &PreparedData) -> Vec<(f64, f64)> {
    let features = data.feature_names.len();
    let mut ranges = vec![(f64::INFINITY, f64::NEG_INFINITY); features];
    let datasets: [Option<&Dataset>; 3] = [
        Some(&data.train),
        data.validation.as_ref(),
        Some(&data.test),
    ];
    for dataset in datasets.into_iter().flatten() {
        let raw = data.input_scaler.inverse_transform(&dataset.inputs);
        for (range, column) in ranges.iter_mut().zip(raw.columns()) {
            for &value in column {
                range.0 = range.0.min(value);
                range.1 = range.1.max(value);
            }
        }
    }
    ranges
}

/// Extends `[min, max]` by `margin` of its width on both sides; a single
/// value gets an interval of width 1 around it.
fn widen(min: f64, max: f64, margin: f64) -> (f64, f64) {
    let width = max - min;
    if width > 0.0 {
        (min - margin * width, max + margin * width)
    } else {
        (min - 0.5, max + 0.5)
    }
}

/// All grid points, one per row: `x` for one axis; for two axes `(x[j], y[i])`
/// at row `i * res_x + j`, so that the outputs reshape to `res_y x res_x`.
fn grid_points(axes: &[Array1<f64>]) -> Array2<f64> {
    match axes {
        [x] => x.clone().insert_axis(ndarray::Axis(1)),
        [x, y] => Array2::from_shape_fn((y.len() * x.len(), 2), |(row, column)| {
            let (i, j) = (row / x.len(), row % x.len());
            if column == 0 { x[j] } else { y[i] }
        }),
        _ => unreachable!("grids have one or two axes"),
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, array};

    use super::*;
    use crate::config::Normalization;
    use crate::data::Scaler;
    use crate::network::{Activation, Layer};

    fn output(resolution: usize, margin: f64) -> OutputConfig {
        OutputConfig {
            grid_resolution: resolution,
            grid_margin: margin,
        }
    }

    /// Data with the given raw inputs, standardized; regression targets 0.
    fn data(train: Array2<f64>, test: Array2<f64>) -> PreparedData {
        let features = train.ncols();
        let input_scaler = Scaler::fit(Normalization::Standard, &train);
        let dataset = |inputs: Array2<f64>| {
            let rows = inputs.nrows();
            Dataset::new(
                input_scaler.transform(&inputs),
                Array2::zeros((rows, 1)),
                None,
            )
        };
        PreparedData {
            train: dataset(train),
            validation: None,
            test: dataset(test),
            feature_names: (0..features).map(|index| format!("x{index}")).collect(),
            target_scaler: Some(Scaler {
                normalization: Normalization::Standard,
                offset: array![100.0],
                scale: array![10.0],
            }),
            input_scaler,
            classes: None,
        }
    }

    /// Network summing its (normalized) inputs.
    fn sum(features: usize) -> Mlp {
        Mlp::from_layers(vec![Layer::from_parts(
            Array2::ones((features, 1)),
            None,
            Activation::Identity,
        )])
    }

    #[test]
    fn one_feature_grid_spans_train_and_test_with_margin() {
        let data = data(array![[0.0], [2.0], [4.0]], array![[6.0]]);
        let grid = Grid::compute(&sum(1), &data, &output(5, 0.25)).unwrap();
        // Range [0, 6] widened by 1.5 on each side.
        let expected = [-1.5, 0.75, 3.0, 5.25, 7.5];
        assert_eq!(grid.axes.len(), 1);
        for (x, expected) in grid.axes[0].iter().zip(expected) {
            assert!(
                (x - expected).abs() < 1e-12,
                "{} != {expected:?}",
                grid.axes[0]
            );
        }
        assert_eq!(grid.outputs.shape(), [5, 1]);
        // Inputs are standardized with mean 2 and std sqrt(8/3), outputs
        // brought back as 10 * z + 100.
        let std = (8.0f64 / 3.0).sqrt();
        for (x, output) in grid.axes[0].iter().zip(grid.outputs.iter()) {
            let expected = 10.0 * (x - 2.0) / std + 100.0;
            assert!((output - expected).abs() < 1e-9, "{output} != {expected}");
        }
    }

    #[test]
    fn two_feature_grid_rows_follow_y() {
        let data = data(array![[0.0, 0.0], [1.0, 10.0]], array![[0.5, 5.0]]);
        let network = Mlp::from_layers(vec![Layer::from_parts(
            array![[1.0], [0.0]],
            None,
            Activation::Identity,
        )]);
        let grid = Grid::compute(&network, &data, &output(3, 0.0)).unwrap();
        assert_eq!(grid.axes, [array![0.0, 0.5, 1.0], array![0.0, 5.0, 10.0]]);
        assert_eq!(grid.outputs.shape(), [3, 3, 1]);
        // The output depends on x only, so it changes along the columns.
        let x = |j: usize| grid.outputs[[0, j, 0]];
        assert!(x(0) < x(1) && x(1) < x(2));
        for i in 1..3 {
            for j in 0..3 {
                assert_eq!(grid.outputs[[i, j, 0]], x(j));
            }
        }
    }

    #[test]
    fn constant_feature_gets_unit_interval() {
        assert_eq!(widen(3.0, 3.0, 0.1), (2.5, 3.5));
    }

    #[test]
    fn no_grid_for_more_than_two_features() {
        let data = data(
            array![[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]],
            array![[0.0, 0.0, 0.0]],
        );
        assert!(Grid::compute(&sum(3), &data, &output(4, 0.1)).is_none());
    }
}
