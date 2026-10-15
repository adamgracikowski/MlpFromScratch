//! Training artifacts: everything the presentation layer needs to plot a
//! run, recorded during training and written to `<experiment>/artifacts/`.

mod loss_history;
mod recorder;
mod weight_history;
mod weight_sample;

pub use loss_history::LossHistory;
pub use recorder::Recorder;
pub use weight_history::WeightHistory;
pub use weight_sample::WeightSample;
