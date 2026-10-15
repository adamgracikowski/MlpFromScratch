"""Network outputs on a dense grid (`artifacts/grid/`)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analysis.experiment.error import load_array


@dataclass(frozen=True)
class Grid:
    """Network outputs on a dense grid spanning the data (1 or 2 input features)."""

    axes: list[np.ndarray]  # [x] or [x, y], each (r,)
    outputs: np.ndarray  # (r, K) or (r, r, K); [i, j] is the point (x[j], y[i])

    @classmethod
    def load(cls, directory: Path, feature_count: int) -> "Grid | None":
        """Load `grid/` of the artifacts; `None` when it is absent (more than 2 features)."""
        if not directory.is_dir():
            return None
        axis_files = ["x.npy", "y.npy"][:feature_count]
        return cls(
            axes=[load_array(directory / name) for name in axis_files],
            outputs=load_array(directory / "outputs.npy"),
        )
