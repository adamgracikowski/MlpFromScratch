//! Training artifacts: everything the presentation layer needs to plot a
//! run, recorded during training and written to `<experiment>/artifacts/`.

mod grid;
mod loss_history;
mod predictions;
mod recorder;
mod weight_history;
mod weight_sample;

pub use grid::Grid;
pub use loss_history::LossHistory;
pub use predictions::{Predictions, SplitPredictions, Targets, predict_in_chunks};
pub use recorder::Recorder;
pub use weight_history::WeightHistory;
pub use weight_sample::WeightSample;
