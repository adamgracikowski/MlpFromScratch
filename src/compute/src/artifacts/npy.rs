//! Arrays in the NumPy `.npy` format, read in Python with `numpy.load`.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use npyz::{AutoSerialize, WriteOptions, WriterBuilder};

use super::ArtifactError;

/// Writes `data` (row-major, C order) as an array of `shape` to `path`.
///
/// Panics when the number of values does not match the shape.
pub(super) fn write_npy<T: AutoSerialize>(
    path: &Path,
    shape: &[usize],
    data: impl IntoIterator<Item = T>,
) -> Result<(), ArtifactError> {
    let error = |source: std::io::Error| ArtifactError::write(path, source);
    let file = File::create(path).map_err(error)?;
    let shape: Vec<u64> = shape.iter().map(|&size| size as u64).collect();
    let mut writer = WriteOptions::new()
        .default_dtype()
        .shape(&shape)
        .writer(BufWriter::new(file))
        .begin_nd()
        .map_err(error)?;
    let mut count = 0u64;
    for value in data {
        writer.push(&value).map_err(error)?;
        count += 1;
    }
    let expected: u64 = shape.iter().product();
    assert_eq!(
        count,
        expected,
        "{}: {count} values for shape {shape:?}",
        path.display()
    );
    writer.finish().map_err(error)
}

/// Reads an array written by [`write_npy`]: its shape and values.
#[cfg(test)]
pub(crate) fn read_npy<T: npyz::Deserialize>(path: &Path) -> (Vec<u64>, Vec<T>) {
    let file = File::open(path).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
    let npy = npyz::NpyFile::new(std::io::BufReader::new(file)).unwrap();
    let shape = npy.shape().to_vec();
    (shape, npy.into_vec().unwrap())
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    #[test]
    fn round_trips_arrays() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("values.npy");
        write_npy(&path, &[2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, f64::NAN]).unwrap();
        let (shape, values) = read_npy::<f64>(&path);
        assert_eq!(shape, [2, 3]);
        assert_eq!(values[..5], [1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(values[5].is_nan());

        write_npy(&path, &[3], [7u64, 8, 9]).unwrap();
        assert_eq!(read_npy::<u64>(&path), (vec![3], vec![7, 8, 9]));
        write_npy(&path, &[0], Vec::<i64>::new()).unwrap();
        assert_eq!(read_npy::<i64>(&path), (vec![0], vec![]));
    }

    #[test]
    #[should_panic(expected = "2 values for shape [3]")]
    fn rejects_wrong_number_of_values() {
        let dir = TempDir::new().unwrap();
        write_npy(&dir.path().join("values.npy"), &[3], [1.0, 2.0]).unwrap();
    }
}
