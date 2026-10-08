//! Settings of the prediction artifacts written after training.

use serde::{Deserialize, Serialize};

/// Settings of the prediction artifacts written after training.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputConfig {
    /// Number of grid points per input dimension (1D and 2D inputs only).
    pub grid_resolution: usize,
    /// Grid extension beyond the data range, as a fraction of that range.
    pub grid_margin: f64,
}
