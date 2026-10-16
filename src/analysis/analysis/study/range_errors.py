"""Test errors of a regression run inside and outside the range of its training inputs.

The test sets of the regression datasets reach beyond the training inputs, so the test MSE
mixes interpolation with extrapolation. A test sample is inside when every feature lies in
the range (minimum to maximum) of that feature over the training samples of the run.
"""

from pathlib import Path

import numpy as np

from analysis.experiment import ARTIFACTS_DIR
from analysis.experiment.error import load_array
from analysis.experiment.predictions import read_inputs
from analysis.study.run_summary import RunSummary

INSIDE = "test_mse_inside"
OUTSIDE = "test_mse_outside"
RANGE_COLUMNS = (INSIDE, OUTSIDE)


class InputCache:
    """Feature columns of dataset CSV files, read once per file and feature list."""

    def __init__(self) -> None:
        self._inputs: dict[tuple[Path, tuple[str, ...]], np.ndarray] = {}

    def get(self, path: Path, features: list[str]) -> np.ndarray:
        key = (path, tuple(features))
        if key not in self._inputs:
            self._inputs[key] = read_inputs(path, features)
        return self._inputs[key]


def range_errors(summary: RunSummary, inputs: InputCache) -> dict[str, float | None]:
    """MSE in the original scale over the test samples inside and outside the training
    range; `None` for a run that did not finish or a part without samples."""
    if not summary.finished:
        return dict.fromkeys(RANGE_COLUMNS)
    data_config = summary.metadata["config"]["data"]
    features = summary.metadata["data"]["features"]
    predictions = summary.directory / ARTIFACTS_DIR / "predictions"

    train_rows = load_array(predictions / "train" / "rows.npy")
    train = inputs.get(Path(data_config["train"]), features)[train_rows]
    test_rows = load_array(predictions / "test" / "rows.npy")
    test = inputs.get(Path(data_config["test"]), features)[test_rows]
    outputs = load_array(predictions / "test" / "outputs.npy")  # (n, 1)
    targets = load_array(predictions / "test" / "targets.npy").ravel()
    errors = (outputs[:, 0] - targets) ** 2

    inside = np.all((test >= train.min(axis=0)) & (test <= train.max(axis=0)), axis=1)
    return {INSIDE: _mean(errors[inside]), OUTSIDE: _mean(errors[~inside])}


def _mean(values: np.ndarray) -> float | None:
    return float(values.mean()) if values.size else None
