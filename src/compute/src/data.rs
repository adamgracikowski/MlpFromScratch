//! Datasets: reading CSV files into matrices and preparing them for training.

mod error;
mod table;

pub use error::DataError;
pub use table::{Table, read_header, read_table};
