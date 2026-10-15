import math

import numpy as np
import pytest

from analysis.series import evenly_spaced, finite_or_nan, moving_average


def test_finite_or_nan() -> None:
    values = np.array([1.0, math.inf, -math.inf, math.nan, 2.0])

    result = finite_or_nan(values)

    np.testing.assert_array_equal(result, [1.0, np.nan, np.nan, np.nan, 2.0])
    assert np.isinf(values[1])  # the input is not modified


def test_moving_average_window_one_is_identity() -> None:
    values = np.array([3.0, 1.0, 2.0])

    np.testing.assert_array_equal(moving_average(values, 1), values)


def test_moving_average_is_trailing() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    np.testing.assert_allclose(moving_average(values, 3), [1.0, 1.5, 2.0, 3.0, 4.0])


def test_moving_average_skips_non_finite() -> None:
    values = np.array([1.0, math.inf, 3.0, math.nan, math.nan, math.nan])

    result = moving_average(values, 2)

    np.testing.assert_allclose(result[:5], [1.0, 1.0, 3.0, 3.0, np.nan])
    assert np.isnan(result[5])


def test_moving_average_rejects_empty_window() -> None:
    with pytest.raises(ValueError, match="window"):
        moving_average(np.ones(3), 0)


def test_evenly_spaced() -> None:
    np.testing.assert_array_equal(evenly_spaced(4, 10), [0, 1, 2, 3])
    indices = evenly_spaced(1001, 11)
    assert len(indices) == 11
    assert indices[0] == 0 and indices[-1] == 1000
    assert np.all(np.diff(indices) > 0)
