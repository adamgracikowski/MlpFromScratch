import numpy as np
import pytest

from analysis.experiment import load_experiment
from analysis.experiment import predictions as predictions_module
from analysis.study import RunSummary
from analysis.study.range_errors import INSIDE, OUTSIDE, InputCache, range_errors


def _expected(experiment_dir) -> tuple[float, float]:
    """The split computed from the loaded experiment, independently of `range_errors`."""
    experiment = load_experiment(experiment_dir)
    train = experiment.predictions["train"].inputs[:, 0]
    test = experiment.predictions["test"]
    errors = (test.outputs[:, 0] - test.targets) ** 2
    inside = (test.inputs[:, 0] >= train.min()) & (test.inputs[:, 0] <= train.max())
    assert inside.any() and not inside.all()
    return errors[inside].mean(), errors[~inside].mean()


def test_mse_inside_and_outside_the_training_range(make_experiment) -> None:
    run = make_experiment(task="regression", features=1)

    errors = range_errors(RunSummary.load(run), InputCache())

    inside, outside = _expected(run)
    assert errors[INSIDE] == pytest.approx(inside)
    assert errors[OUTSIDE] == pytest.approx(outside)


def test_diverged_run_has_no_errors(make_experiment) -> None:
    run = make_experiment(task="regression", features=1, diverged=True)

    assert range_errors(RunSummary.load(run), InputCache()) == {INSIDE: None, OUTSIDE: None}


def test_inputs_read_once_per_file(make_experiment, monkeypatch) -> None:
    reads = []
    original = predictions_module.read_inputs

    def counting(path, features):
        reads.append(path)
        return original(path, features)

    monkeypatch.setattr("analysis.study.range_errors.read_inputs", counting)
    cache = InputCache()
    run = make_experiment(task="regression", features=1)
    for _ in range(3):
        range_errors(RunSummary.load(run), cache)

    assert len(reads) == 2  # train and test file
    assert isinstance(cache.get(reads[0], ["x"]), np.ndarray)
