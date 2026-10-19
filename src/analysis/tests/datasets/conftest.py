import gzip
import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest

from analysis.datasets import IdxFile, MnistSplit

_TYPE_CODES = {np.dtype(np.uint8): 0x08, np.dtype(">i2"): 0x0B}


def _idx_bytes(array: np.ndarray) -> bytes:
    """Contents of an IDX file holding `array` (uint8 or big-endian int16)."""
    header = bytes([0, 0, _TYPE_CODES[array.dtype], array.ndim])
    return header + np.array(array.shape, ">u4").tobytes() + array.tobytes()


@pytest.fixture
def make_idx() -> Callable[[np.ndarray], bytes]:
    """Contents of an IDX file holding an array (uint8 or big-endian int16)."""
    return _idx_bytes


@dataclass(frozen=True)
class SmallMnist:
    """A tiny MNIST-like dataset: 2x2 images, 3 training and 2 test samples."""

    splits: tuple[MnistSplit, ...]
    files: dict[str, bytes]  # IDX contents by file name
    images: dict[str, np.ndarray]  # by split name
    labels: dict[str, np.ndarray]

    def fetch(self, url: str) -> bytes:
        """Stands in for the download: the gzipped file named by the URL."""
        name = url.rsplit("/", 1)[1].removesuffix(".gz")
        return gzip.compress(self.files[name], mtime=0)


@pytest.fixture
def small_mnist() -> SmallMnist:
    files: dict[str, bytes] = {}
    splits = []
    images = {}
    labels = {}
    for name, prefix, count in [("train", "train", 3), ("test", "t10k", 2)]:
        images[name] = np.arange(count * 4, dtype=np.uint8).reshape(count, 2, 2) * 20
        labels[name] = np.arange(count, dtype=np.uint8)[::-1].copy()
        idx_files = []
        for kind, array in [("images-idx3", images[name]), ("labels-idx1", labels[name])]:
            file_name = f"{prefix}-{kind}-ubyte"
            files[file_name] = _idx_bytes(array)
            digest = hashlib.sha256(files[file_name]).hexdigest()
            idx_files.append(IdxFile(file_name, digest, array.shape))
        splits.append(MnistSplit(name, *idx_files, csv=f"data.mnist.{name}.{count}.csv"))
    return SmallMnist(tuple(splits), files, images, labels)
