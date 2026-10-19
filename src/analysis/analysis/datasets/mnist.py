"""MNIST: download (or take a local copy), verification and conversion to CSV.

The four IDX files are kept in `<output>/raw/` and checked against their SHA-256, so a copy
from the Kaggle archive and a download from the mirror are byte-identical. Every split becomes
one CSV file read by the compute layer: a `label` column (0-9) and the raw pixel values
(0-255) in columns `pixel0` to `pixel783`, row by row of the 28 x 28 image.
"""

import gzip
import hashlib
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import numpy as np
import polars as pl

from analysis.datasets.error import DatasetError
from analysis.datasets.idx import read_idx

MIRROR_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
RAW_DIR = "raw"
LABEL_COLUMN = "label"
DOWNLOAD_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class IdxFile:
    """One IDX file of the dataset."""

    name: str  # file name in `raw/`; on the mirror with `.gz` appended
    sha256: str  # of the uncompressed file
    shape: tuple[int, ...]

    @property
    def names(self) -> tuple[str, ...]:
        """Names the file is looked up by in a local source: the original one, the one of the
        Kaggle archive (`train-images.idx3-ubyte`) and their gzipped variants."""
        names = (self.name, self.name.replace("-idx", ".idx"))
        return names + tuple(f"{name}.gz" for name in names)


@dataclass(frozen=True)
class MnistSplit:
    """Images and labels of one split and the CSV file they are converted to."""

    name: str
    images: IdxFile
    labels: IdxFile
    csv: str


SPLITS = (
    MnistSplit(
        name="train",
        images=IdxFile(
            "train-images-idx3-ubyte",
            "ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db",
            (60000, 28, 28),
        ),
        labels=IdxFile(
            "train-labels-idx1-ubyte",
            "65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5",
            (60000,),
        ),
        csv="data.mnist.train.60000.csv",
    ),
    MnistSplit(
        name="test",
        images=IdxFile(
            "t10k-images-idx3-ubyte",
            "0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7",
            (10000, 28, 28),
        ),
        labels=IdxFile(
            "t10k-labels-idx1-ubyte",
            "ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2",
            (10000,),
        ),
        csv="data.mnist.test.10000.csv",
    ),
)


@dataclass
class MnistReport:
    """What happened to every file, in order: (path, description)."""

    files: list[tuple[Path, str]] = field(default_factory=list)


def download(url: str) -> bytes:
    """Download a file from the mirror."""
    try:
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            return response.read()
    except (OSError, ValueError) as error:
        raise DatasetError(
            f"cannot download {url}: {error}; download the archive from Kaggle "
            "(see data/datasets/mnist/README.md) and pass it with --source"
        ) from error


def prepare_mnist(
    output_dir: Path,
    source: Path | None = None,
    force: bool = False,
    splits: tuple[MnistSplit, ...] = SPLITS,
    fetch: Callable[[str], bytes] = download,
) -> MnistReport:
    """Put the verified IDX files into `<output_dir>/raw/` and write the CSV file of every split.

    A raw file that is present and passes its checksum is kept; otherwise it is taken from
    `source` (a directory or a ZIP archive, e.g. the one from Kaggle) or downloaded from the
    mirror. A CSV file is written when it is missing, when its raw files changed or with `force`.
    """
    report = MnistReport()
    raw_dir = output_dir / RAW_DIR
    for split in splits:
        changed = [
            _ensure_raw(idx_file, raw_dir, source, fetch, report)
            for idx_file in (split.images, split.labels)
        ]
        csv_path = output_dir / split.csv
        if csv_path.is_file() and not force and not any(changed):
            report.files.append((csv_path, "present"))
            continue
        rows = _write_csv(split, raw_dir, csv_path)
        report.files.append((csv_path, f"written, {rows} rows"))
    return report


def _ensure_raw(
    idx_file: IdxFile,
    raw_dir: Path,
    source: Path | None,
    fetch: Callable[[str], bytes],
    report: MnistReport,
) -> bool:
    """Make `raw_dir/<name>` hold the verified file; `True` when it was (re)written."""
    path = raw_dir / idx_file.name
    if path.is_file() and _sha256(path.read_bytes()) == idx_file.sha256:
        report.files.append((path, "verified"))
        return False
    if source is None:
        url = f"{MIRROR_URL}{idx_file.name}.gz"
        data = _decompress(fetch(url), url)
        origin = f"downloaded from {url}"
    else:
        data, location = _find_in_source(idx_file, source)
        origin = f"taken from {location}"
    actual = _sha256(data)
    if actual != idx_file.sha256:
        raise DatasetError(
            f"{idx_file.name} {origin} has SHA-256 {actual}, expected {idx_file.sha256}: "
            "the file is damaged or not the original MNIST"
        )
    _write_bytes(path, data)
    report.files.append((path, origin))
    return True


def _find_in_source(idx_file: IdxFile, source: Path) -> tuple[bytes, str]:
    """Contents of the file found by name in a directory or a ZIP archive (at any depth)
    and where it was found."""
    if source.is_dir():
        found = sorted(p for p in source.rglob("*") if p.is_file() and p.name in idx_file.names)
        if found:
            location = found[0].as_posix()
            return _decompress(found[0].read_bytes(), location), location
    elif zipfile.is_zipfile(source):
        try:
            with zipfile.ZipFile(source) as archive:
                members = sorted(
                    member.filename
                    for member in archive.infolist()
                    if not member.is_dir() and PurePosixPath(member.filename).name in idx_file.names
                )
                if members:
                    location = f"{source.as_posix()}:{members[0]}"
                    return _decompress(archive.read(members[0]), location), location
        except (OSError, zipfile.BadZipFile) as error:
            raise DatasetError(f"cannot read {source.as_posix()}: {error}") from error
    else:
        raise DatasetError(f"source {source.as_posix()} is neither a directory nor a ZIP archive")
    raise DatasetError(
        f"{idx_file.name} not found in {source.as_posix()} (looked for {', '.join(idx_file.names)})"
    )


def _decompress(data: bytes, location: str) -> bytes:
    """Uncompress gzipped data (recognized by its magic bytes), pass other data through."""
    if not data.startswith(b"\x1f\x8b"):
        return data
    try:
        return gzip.decompress(data)
    except (OSError, EOFError) as error:
        raise DatasetError(f"cannot decompress {location}: {error}") from error


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_csv(split: MnistSplit, raw_dir: Path, path: Path) -> int:
    """Write the CSV file of a split; returns the number of rows."""
    images = read_idx(raw_dir / split.images.name)
    labels = read_idx(raw_dir / split.labels.name)
    if images.shape != split.images.shape or labels.shape != split.labels.shape:
        raise DatasetError(
            f"{split.name} images and labels have shapes {images.shape} and {labels.shape}, "
            f"expected {split.images.shape} and {split.labels.shape}"
        )
    pixels = images.reshape(len(labels), -1)
    columns = [LABEL_COLUMN, *(f"pixel{index}" for index in range(pixels.shape[1]))]
    table = pl.DataFrame(np.column_stack([labels, pixels]), schema=columns, orient="row")
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        table.write_csv(temporary)
        temporary.replace(path)
    except OSError as error:
        raise DatasetError(f"cannot write {path.as_posix()}: {error}") from error
    return table.height


def _write_bytes(path: Path, data: bytes) -> None:
    """Write through a temporary file, so an interrupted run leaves no partial file."""
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(data)
        temporary.replace(path)
    except OSError as error:
        raise DatasetError(f"cannot write {path.as_posix()}: {error}") from error
