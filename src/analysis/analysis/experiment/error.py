"""Errors of reading artifacts, and the array loader that reports them."""

from pathlib import Path

import numpy as np


class ArtifactsError(Exception):
    """Artifacts of an experiment are missing or cannot be read."""


def load_array(path: Path) -> np.ndarray:
    """Load a `.npy` file, reporting a missing or broken file as `ArtifactsError`."""
    try:
        return np.load(path)
    except (OSError, ValueError) as error:
        raise ArtifactsError(f"cannot read {path.as_posix()}: {error}") from error
