"""Parameters of the trained network (`artifacts/model/`)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analysis.experiment.error import load_array


@dataclass(frozen=True)
class Model:
    """Weights and biases of the network returned by training (with early stopping: of
    the best epoch)."""

    weights: list[np.ndarray]  # per layer (in, out)
    biases: list[np.ndarray | None]  # per layer (out,), None without bias

    @classmethod
    def load(cls, directory: Path, layer_count: int, bias: bool) -> "Model":
        """Load `model/` of the artifacts."""
        layers = range(layer_count)
        return cls(
            weights=[load_array(directory / f"layer_{layer}_weights.npy") for layer in layers],
            biases=[
                load_array(directory / f"layer_{layer}_bias.npy") if bias else None
                for layer in layers
            ],
        )
