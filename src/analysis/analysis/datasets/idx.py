"""Reading the IDX format of MNIST (http://yann.lecun.com/exdb/mnist/).

An IDX file starts with a magic number: two zero bytes, a byte with the element type and a
byte with the number of dimensions. The size of every dimension follows as a big-endian
32-bit unsigned integer, then the elements in row-major order (multi-byte types big-endian).
"""

from pathlib import Path

import numpy as np

from analysis.datasets.error import DatasetError

_HEADER_BYTES = 4
_DIMENSION_BYTES = 4

# Element type codes of the IDX format.
_TYPES = {
    0x08: np.dtype(np.uint8),
    0x09: np.dtype(np.int8),
    0x0B: np.dtype(">i2"),
    0x0C: np.dtype(">i4"),
    0x0D: np.dtype(">f4"),
    0x0E: np.dtype(">f8"),
}


def parse_idx(data: bytes, name: str) -> np.ndarray:
    """Parse the contents of an IDX file; `name` identifies the file in error messages."""
    if len(data) < _HEADER_BYTES or data[0] != 0 or data[1] != 0 or data[2] not in _TYPES:
        raise DatasetError(f"{name} is not an IDX file: unknown magic number {data[:4].hex()}")
    dtype = _TYPES[data[2]]
    dimensions = data[3]
    offset = _HEADER_BYTES + _DIMENSION_BYTES * dimensions
    if len(data) < offset:
        raise DatasetError(f"{name} is truncated: the header lists {dimensions} dimensions")
    shape = tuple(int(size) for size in np.frombuffer(data, ">u4", dimensions, _HEADER_BYTES))
    expected = int(np.prod(shape)) * dtype.itemsize
    if len(data) - offset != expected:
        raise DatasetError(
            f"{name} has {len(data) - offset} bytes of data, expected {expected} for shape {shape}"
        )
    return np.frombuffer(data, dtype, offset=offset).reshape(shape)


def read_idx(path: Path) -> np.ndarray:
    """Read an IDX file into an array of its shape and element type."""
    try:
        data = path.read_bytes()
    except OSError as error:
        raise DatasetError(f"cannot read {path.as_posix()}: {error}") from error
    return parse_idx(data, path.as_posix())
