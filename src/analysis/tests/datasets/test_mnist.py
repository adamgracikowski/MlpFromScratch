import gzip
import zipfile

import numpy as np
import polars as pl
import pytest

from analysis.datasets import MIRROR_URL, SPLITS, DatasetError, prepare_mnist


def _no_download(url: str) -> bytes:
    raise AssertionError(f"unexpected download of {url}")


def _kaggle_name(name: str) -> str:
    return name.replace("-idx", ".idx")


def test_downloads_and_writes_csv(small_mnist, tmp_path) -> None:
    urls = []

    def fetch(url: str) -> bytes:
        urls.append(url)
        return small_mnist.fetch(url)

    report = prepare_mnist(tmp_path, splits=small_mnist.splits, fetch=fetch)

    assert urls == [f"{MIRROR_URL}{name}.gz" for name in small_mnist.files]
    for name, data in small_mnist.files.items():
        assert (tmp_path / "raw" / name).read_bytes() == data
    train = pl.read_csv(tmp_path / "data.mnist.train.3.csv")
    assert train.columns == ["label", "pixel0", "pixel1", "pixel2", "pixel3"]
    np.testing.assert_array_equal(train["label"], small_mnist.labels["train"])
    np.testing.assert_array_equal(
        train.drop("label").to_numpy(), small_mnist.images["train"].reshape(3, 4)
    )
    assert pl.read_csv(tmp_path / "data.mnist.test.2.csv").height == 2
    descriptions = [description for _, description in report.files]
    assert descriptions[0].startswith("downloaded from")
    assert descriptions[2] == "written, 3 rows"
    assert not list(tmp_path.rglob("*.tmp"))


def test_keeps_verified_files(small_mnist, tmp_path) -> None:
    prepare_mnist(tmp_path, splits=small_mnist.splits, fetch=small_mnist.fetch)

    report = prepare_mnist(tmp_path, splits=small_mnist.splits, fetch=_no_download)

    descriptions = [description for _, description in report.files]
    assert descriptions == ["verified", "verified", "present"] * 2


def test_force_rewrites_csv_without_download(small_mnist, tmp_path) -> None:
    prepare_mnist(tmp_path, splits=small_mnist.splits, fetch=small_mnist.fetch)
    (tmp_path / "data.mnist.train.3.csv").write_text("broken")

    report = prepare_mnist(tmp_path, splits=small_mnist.splits, force=True, fetch=_no_download)

    assert report.files[2][1] == "written, 3 rows"
    assert pl.read_csv(tmp_path / "data.mnist.train.3.csv").height == 3


def test_replaces_damaged_raw_file(small_mnist, tmp_path) -> None:
    prepare_mnist(tmp_path, splits=small_mnist.splits, fetch=small_mnist.fetch)
    (tmp_path / "raw" / "t10k-labels-idx1-ubyte").write_bytes(b"damaged")

    report = prepare_mnist(tmp_path, splits=small_mnist.splits, fetch=small_mnist.fetch)

    descriptions = [description for _, description in report.files]
    assert descriptions[2] == "present"
    assert descriptions[4].startswith("downloaded from")
    assert descriptions[5] == "written, 2 rows"


def test_takes_files_from_kaggle_directory(small_mnist, tmp_path) -> None:
    source = tmp_path / "archive"
    for name, data in small_mnist.files.items():
        nested = source / name / name
        nested.parent.mkdir(parents=True)
        nested.write_bytes(data)

    report = prepare_mnist(
        tmp_path / "mnist", source=source, splits=small_mnist.splits, fetch=_no_download
    )

    nested = source / "train-images-idx3-ubyte" / "train-images-idx3-ubyte"
    assert report.files[0][1] == f"taken from {nested.as_posix()}"
    assert (tmp_path / "mnist" / "data.mnist.test.2.csv").is_file()


def test_takes_files_from_kaggle_zip(small_mnist, tmp_path) -> None:
    source = tmp_path / "archive.zip"
    with zipfile.ZipFile(source, "w") as archive:
        for name, data in small_mnist.files.items():
            archive.writestr(_kaggle_name(name), data)
            archive.writestr(f"{name}/{name}", data)

    report = prepare_mnist(
        tmp_path / "mnist", source=source, splits=small_mnist.splits, fetch=_no_download
    )

    member = "train-images-idx3-ubyte/train-images-idx3-ubyte"  # the first one by name
    assert report.files[0][1] == f"taken from {source.as_posix()}:{member}"
    for name, data in small_mnist.files.items():
        assert (tmp_path / "mnist" / "raw" / name).read_bytes() == data


def test_takes_gzipped_files_from_directory(small_mnist, tmp_path) -> None:
    source = tmp_path / "archive"
    source.mkdir()
    for name, data in small_mnist.files.items():
        (source / f"{name}.gz").write_bytes(gzip.compress(data))

    prepare_mnist(tmp_path / "mnist", source=source, splits=small_mnist.splits)

    for name, data in small_mnist.files.items():
        assert (tmp_path / "mnist" / "raw" / name).read_bytes() == data


def test_rejects_wrong_checksum(small_mnist, tmp_path) -> None:
    def fetch(url: str) -> bytes:
        return gzip.compress(small_mnist.files["train-images-idx3-ubyte"][:-1] + b"\x01")

    with pytest.raises(DatasetError, match="not the original MNIST"):
        prepare_mnist(tmp_path, splits=small_mnist.splits, fetch=fetch)

    assert not (tmp_path / "raw").exists()


def test_reports_file_missing_from_source(small_mnist, tmp_path) -> None:
    source = tmp_path / "archive"
    source.mkdir()

    with pytest.raises(DatasetError, match="train-images.idx3-ubyte"):
        prepare_mnist(tmp_path / "mnist", source=source, splits=small_mnist.splits)


def test_rejects_source_of_other_kind(small_mnist, tmp_path) -> None:
    source = tmp_path / "archive.txt"
    source.write_text("text")

    with pytest.raises(DatasetError, match="neither a directory nor a ZIP"):
        prepare_mnist(tmp_path / "mnist", source=source, splits=small_mnist.splits)


def test_splits_match_mnist() -> None:
    train, test = SPLITS

    assert train.images.shape == (60000, 28, 28)
    assert test.labels.shape == (10000,)
    assert train.csv == "data.mnist.train.60000.csv"
    assert test.images.names == (
        "t10k-images-idx3-ubyte",
        "t10k-images.idx3-ubyte",
        "t10k-images-idx3-ubyte.gz",
        "t10k-images.idx3-ubyte.gz",
    )
