"""Datasets that are not in the repository: download, verification and conversion to CSV."""

from analysis.datasets.error import DatasetError
from analysis.datasets.idx import parse_idx, read_idx
from analysis.datasets.mnist import (
    MIRROR_URL,
    SPLITS,
    IdxFile,
    MnistReport,
    MnistSplit,
    prepare_mnist,
)

__all__ = [
    "MIRROR_URL",
    "SPLITS",
    "DatasetError",
    "IdxFile",
    "MnistReport",
    "MnistSplit",
    "parse_idx",
    "prepare_mnist",
    "read_idx",
]
