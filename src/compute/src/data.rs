//! Datasets: reading CSV files into matrices and preparing them for training.

mod classes;
mod error;
mod scaler;
mod split;
mod table;

pub use classes::{ClassEncoding, MIN_CLASSES};
pub use error::DataError;
pub use scaler::Scaler;
pub use split::{Split, split_indices, stratified_quotas};
pub use table::{Table, read_header, read_table};
