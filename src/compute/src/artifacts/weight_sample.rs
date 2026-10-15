//! The fixed subset of parameters whose values are recorded during training.

use rand::Rng;
use rand::seq::index;

use crate::network::{Layer, Mlp};

/// Parameters of every layer recorded in the weight history.
///
/// The parameters of a layer form the augmented matrix `[W; b]` of shape
/// `(in + 1) x out`: row `in` holds the bias (without bias the matrix is
/// just `W`, `in x out`). A parameter is identified by its `(row, column)`
/// in that matrix. When a layer has more than `max_per_layer` parameters, a
/// random subset of that size is drawn once and kept for the whole run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightSample {
    /// Recorded `(row, column)` pairs of every layer, in row-major order.
    layers: Vec<Vec<(usize, usize)>>,
}

impl WeightSample {
    /// Draws at most `max_per_layer` parameters of every layer of `network`
    /// (`max_per_layer >= 1`).
    pub fn draw<R: Rng + ?Sized>(network: &Mlp, max_per_layer: usize, rng: &mut R) -> Self {
        assert!(
            max_per_layer > 0,
            "at least one weight per layer is recorded"
        );
        let layers = network
            .layers()
            .iter()
            .map(|layer| {
                let rows = layer.input_size() + usize::from(layer.bias().is_some());
                let columns = layer.output_size();
                let pool = rows * columns;
                let mut flat = if pool <= max_per_layer {
                    (0..pool).collect()
                } else {
                    index::sample(rng, pool, max_per_layer).into_vec()
                };
                flat.sort_unstable();
                flat.into_iter()
                    .map(|index| (index / columns, index % columns))
                    .collect()
            })
            .collect();
        Self { layers }
    }

    /// Recorded `(row, column)` pairs of every layer.
    pub fn layers(&self) -> &[Vec<(usize, usize)>] {
        &self.layers
    }

    /// Appends the current values of the recorded parameters of `layer`
    /// (number `index` in the network) to `values`.
    pub fn read(&self, index: usize, layer: &Layer, values: &mut Vec<f64>) {
        let weights = layer.weights();
        let inputs = layer.input_size();
        values.extend(self.layers[index].iter().map(|&(row, column)| {
            if row == inputs {
                layer.bias().expect("row `in` is the bias")[column]
            } else {
                weights[[row, column]]
            }
        }));
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;
    use crate::network::Activation;
    use crate::rng::{RngStream, rng};

    /// 2 -> 3 (bias) -> 1 (no bias): 9 and 3 parameters.
    fn network() -> Mlp {
        Mlp::from_layers(vec![
            Layer::from_parts(
                array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                Some(array![7.0, 8.0, 9.0]),
                Activation::Tanh,
            ),
            Layer::from_parts(array![[10.0], [11.0], [12.0]], None, Activation::Identity),
        ])
    }

    fn draw(max: usize, seed: u64) -> WeightSample {
        WeightSample::draw(&network(), max, &mut rng(seed, RngStream::WeightLog))
    }

    #[test]
    fn small_layers_are_recorded_whole() {
        let sample = draw(9, 1);
        let first: Vec<(usize, usize)> = (0..3)
            .flat_map(|row| (0..3).map(move |column| (row, column)))
            .collect();
        assert_eq!(sample.layers(), [first, vec![(0, 0), (1, 0), (2, 0)]]);

        let mut values = Vec::new();
        sample.read(0, &network().layers()[0], &mut values);
        // Row 2 of [W; b] is the bias.
        assert_eq!(values, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        values.clear();
        sample.read(1, &network().layers()[1], &mut values);
        assert_eq!(values, [10.0, 11.0, 12.0]);
    }

    #[test]
    fn large_layers_are_subsampled_in_order() {
        let sample = draw(4, 1);
        let first = &sample.layers()[0];
        assert_eq!(first.len(), 4);
        assert!(first.windows(2).all(|pair| pair[0] < pair[1]), "{first:?}");
        assert!(first.iter().all(|&(row, column)| row < 3 && column < 3));
        assert_eq!(sample.layers()[1].len(), 3);
    }

    #[test]
    fn subset_depends_on_seed_only() {
        assert_eq!(draw(4, 1), draw(4, 1));
        let others = (2..10).filter(|&seed| draw(4, seed) != draw(4, 1)).count();
        assert!(others > 0);
    }
}
