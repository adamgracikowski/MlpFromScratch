import numpy as np
import pytest

from analysis.experiment import ArtifactsError, LossHistory


def test_loads_history(make_experiment) -> None:
    history = LossHistory.load(make_experiment() / "artifacts/history")

    assert history.iteration.shape == history.epoch.shape == history.loss.shape
    assert history.iteration.dtype == np.uint64
    assert history.loss.dtype == np.float64
    assert history.iteration[0] == 1


def test_keeps_non_finite_loss(make_experiment) -> None:
    history = LossHistory.load(make_experiment(diverged=True) / "artifacts/history")

    assert np.isinf(history.loss[-2])
    assert np.isnan(history.loss[-1])


def test_missing_file(make_experiment) -> None:
    directory = make_experiment() / "artifacts/history"
    (directory / "loss.npy").unlink()

    with pytest.raises(ArtifactsError, match="loss.npy"):
        LossHistory.load(directory)
