"""Predictions of the trained network (`artifacts/predictions/<split>/`)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from analysis.experiment.error import ArtifactsError, load_array

SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitPredictions:
    """Predictions of the trained network on one split, with the inputs read from the CSV."""

    rows: np.ndarray  # (n,) row in the CSV file (0 = first data row)
    inputs: np.ndarray  # (n, d) in the original scale
    outputs: np.ndarray  # (n, K) class probabilities or predictions in the original scale
    labels: np.ndarray | None  # (n,) true class labels (classification)
    predicted: np.ndarray | None  # (n,) predicted class labels (classification)
    targets: np.ndarray | None  # (n,) true targets in the original scale (regression)

    @classmethod
    def load_all(cls, directory: Path, metadata: dict[str, Any]) -> dict[str, "SplitPredictions"]:
        """Load `predictions/` for every split present, keyed in `SPLITS` order."""
        data_config = metadata["config"]["data"]
        features = metadata["data"]["features"]
        classification = metadata["config"]["task"] == "classification"
        # Validation samples are held out from the training file.
        inputs = {
            "train": read_inputs(Path(data_config["train"]), features),
            "test": read_inputs(Path(data_config["test"]), features),
        }
        inputs["validation"] = inputs["train"]

        predictions = {}
        for split in SPLITS:
            split_dir = directory / split
            if not split_dir.is_dir():
                continue
            rows = load_array(split_dir / "rows.npy")
            if rows.size and rows.max() >= len(inputs[split]):
                raise ArtifactsError(
                    f"{(split_dir / 'rows.npy').as_posix()} points past the end of the CSV "
                    "file: the dataset changed since training, run `compute train` again"
                )
            targets = None if classification else load_array(split_dir / "targets.npy").ravel()
            predictions[split] = cls(
                rows=rows,
                inputs=inputs[split][rows],
                outputs=load_array(split_dir / "outputs.npy"),
                labels=load_array(split_dir / "labels.npy") if classification else None,
                predicted=load_array(split_dir / "predicted.npy") if classification else None,
                targets=targets,
            )
        return predictions


def read_inputs(path: Path, features: list[str]) -> np.ndarray:
    """Read the feature columns of a dataset CSV as an (n, d) float array."""
    try:
        frame = pl.read_csv(path, columns=features)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise ArtifactsError(
            f"cannot read the inputs from {path.as_posix()} (paths in the config are relative "
            f"to the current working directory, run from the repository root): {error}"
        ) from error
    return frame.select(features).to_numpy().astype(np.float64)
