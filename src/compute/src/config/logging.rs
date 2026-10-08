//! How often training progress is recorded.

use serde::{Deserialize, Serialize};

/// How often training progress is recorded, counted in iterations (batches).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoggingConfig {
    /// Interval between recorded loss values.
    pub loss_every: usize,
    /// Interval between weight snapshots.
    pub weights_every: usize,
    /// Upper bound on the weights recorded per layer; a fixed random subset
    /// is drawn from the seed when a layer has more.
    pub max_weights_per_layer: usize,
}
