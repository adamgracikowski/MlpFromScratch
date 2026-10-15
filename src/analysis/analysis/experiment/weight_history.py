"""Evolution of the weights (`artifacts/weights/`)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analysis.experiment.error import load_array


@dataclass(frozen=True)
class WeightHistory:
    """Snapshots of the sampled weights and the norms of every layer."""

    iteration: np.ndarray  # (S,) 0 = initial weights
    indices: list[np.ndarray]  # per layer (m, 2): (row, column) of [W; b], row `in` is the bias
    values: list[np.ndarray]  # per layer (S, m)
    weight_norms: np.ndarray  # (S, L) Frobenius norm of W
    bias_norms: np.ndarray  # (S, L) Euclidean norm of b, NaN without bias

    @property
    def layer_count(self) -> int:
        return len(self.values)

    @classmethod
    def load(cls, directory: Path, layer_count: int) -> "WeightHistory":
        """Load `weights/` of the artifacts for a network of `layer_count` layers."""
        layers = range(layer_count)
        return cls(
            iteration=load_array(directory / "iteration.npy"),
            indices=[load_array(directory / f"layer_{layer}_indices.npy") for layer in layers],
            values=[load_array(directory / f"layer_{layer}_values.npy") for layer in layers],
            weight_norms=load_array(directory / "weight_norms.npy"),
            bias_norms=load_array(directory / "bias_norms.npy"),
        )
