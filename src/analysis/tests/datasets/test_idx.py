import numpy as np
import pytest

from analysis.datasets import DatasetError, parse_idx, read_idx


def test_parses_unsigned_bytes(make_idx) -> None:
    array = np.arange(12, dtype=np.uint8).reshape(3, 2, 2)

    parsed = parse_idx(make_idx(array), "images")

    assert parsed.dtype == np.uint8
    np.testing.assert_array_equal(parsed, array)


def test_parses_big_endian_integers(make_idx) -> None:
    array = np.array([-2, 300, 1000], dtype=">i2")

    np.testing.assert_array_equal(parse_idx(make_idx(array), "values"), [-2, 300, 1000])


def test_rejects_unknown_magic_number() -> None:
    with pytest.raises(DatasetError, match="not an IDX file"):
        parse_idx(b"PK\x03\x04rest", "archive.zip")


def test_rejects_missing_data(make_idx) -> None:
    data = make_idx(np.zeros((3, 2), dtype=np.uint8))

    with pytest.raises(DatasetError, match="5 bytes of data, expected 6"):
        parse_idx(data[:-1], "labels")


def test_rejects_truncated_header() -> None:
    with pytest.raises(DatasetError, match="truncated"):
        parse_idx(bytes([0, 0, 0x08, 3, 0, 0, 0, 1]), "images")


def test_reports_missing_file(tmp_path) -> None:
    with pytest.raises(DatasetError, match="cannot read"):
        read_idx(tmp_path / "missing")
