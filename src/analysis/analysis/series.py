"""Helpers for plotted series: non-finite values and smoothing."""

import numpy as np


def finite_or_nan(values: np.ndarray) -> np.ndarray:
    """Copy as float with `inf` replaced by NaN, so plots leave a gap after divergence."""
    result = np.asarray(values, dtype=np.float64).copy()
    result[~np.isfinite(result)] = np.nan
    return result


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean over the last `window` values, skipping non-finite ones.

    The first `window - 1` points average the values available so far; a point whose
    window holds no finite value is NaN.
    """
    if window < 1:
        raise ValueError(f"window must be positive, got {window}")
    values = finite_or_nan(values)
    finite = ~np.isnan(values)
    sums = np.cumsum(np.where(finite, values, 0.0))
    counts = np.cumsum(finite)
    sums[window:] = sums[window:] - sums[:-window]
    counts[window:] = counts[window:] - counts[:-window]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def evenly_spaced(count: int, limit: int) -> np.ndarray:
    """Indices of at most `limit` evenly spaced items out of `count`, first and last included."""
    if count <= limit:
        return np.arange(count)
    return np.unique(np.linspace(0, count - 1, limit).round().astype(np.int64))
