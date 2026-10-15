import math

import numpy as np

from analysis.plots.study_curves import mean_and_sd


def test_mean_and_sd_of_series_with_different_lengths() -> None:
    epochs, mean, sd = mean_and_sd([np.array([1.0, 2.0, 3.0]), np.array([3.0, 4.0])])

    assert epochs.tolist() == [0, 1, 2]
    assert mean.tolist() == [2.0, 3.0, 3.0]
    assert sd[0] == sd[1] == math.sqrt(2)
    assert math.isnan(sd[2])


def test_mean_and_sd_ignore_missing_values() -> None:
    _, mean, sd = mean_and_sd([np.array([1.0, np.nan]), np.array([3.0, np.nan])])

    assert mean[0] == 2.0
    assert math.isnan(mean[1])
    assert math.isnan(sd[1])
