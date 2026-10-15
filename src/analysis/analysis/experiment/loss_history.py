"""Loss during training (`artifacts/history/`)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analysis.experiment.error import load_array


@dataclass(frozen=True)
class LossHistory:
    """Mean batch loss of every `logging.loss_every` window."""

    iteration: np.ndarray  # (n,) last iteration of the window
    epoch: np.ndarray  # (n,) epoch of that iteration
    loss: np.ndarray  # (n,)

    @classmethod
    def load(cls, directory: Path) -> "LossHistory":
        """Load `history/` of the artifacts."""
        return cls(
            iteration=load_array(directory / "iteration.npy"),
            epoch=load_array(directory / "epoch.npy"),
            loss=load_array(directory / "loss.npy"),
        )
