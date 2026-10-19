"""All artifacts of one experiment and their loader."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from analysis.experiment.error import ArtifactsError
from analysis.experiment.grid import Grid
from analysis.experiment.loss_history import LossHistory
from analysis.experiment.model import Model
from analysis.experiment.predictions import SplitPredictions
from analysis.experiment.weight_history import WeightHistory

ARTIFACTS_DIR = "artifacts"
MIN_IMAGE_SIDE = 8


@dataclass(frozen=True)
class Experiment:
    """All artifacts of one experiment."""

    directory: Path
    metadata: dict[str, Any]
    epochs: pl.DataFrame
    history: LossHistory
    weights: WeightHistory
    model: Model
    predictions: dict[str, SplitPredictions]  # keys in `SPLITS` order, validation only if any
    grid: Grid | None

    @property
    def task(self) -> str:
        return self.metadata["config"]["task"]

    @property
    def is_classification(self) -> bool:
        return self.task == "classification"

    @property
    def features(self) -> list[str]:
        return self.metadata["data"]["features"]

    @property
    def image_side(self) -> int | None:
        """Side of the square images whose pixels are the input features (`d = side^2`,
        e.g. 28 for MNIST), or `None` when the features are not an image."""
        side = math.isqrt(len(self.features))
        return side if side >= MIN_IMAGE_SIDE and side * side == len(self.features) else None

    @property
    def classes(self) -> list[int] | None:
        return self.metadata["data"]["classes"]

    @property
    def best_epoch(self) -> int | None:
        return self.metadata["result"]["best_epoch"]

    @property
    def stop_reason(self) -> str:
        return self.metadata["result"]["stop"]["reason"]


def load_experiment(experiment_dir: Path) -> Experiment:
    """Load `<experiment_dir>/artifacts/` and the inputs of every predicted sample."""
    artifacts = experiment_dir / ARTIFACTS_DIR
    metadata_path = artifacts / "metadata.json"
    if not metadata_path.is_file():
        raise ArtifactsError(
            f"no artifacts in {artifacts.as_posix()}: "
            f"run `compute train {experiment_dir.as_posix()}` first"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    layer_count = len(metadata["network"]["layer_sizes"]) - 1
    return Experiment(
        directory=experiment_dir,
        metadata=metadata,
        epochs=pl.read_csv(artifacts / "epochs.csv"),
        history=LossHistory.load(artifacts / "history"),
        weights=WeightHistory.load(artifacts / "weights", layer_count),
        model=Model.load(artifacts / "model", layer_count, metadata["network"]["bias"]),
        predictions=SplitPredictions.load_all(artifacts / "predictions", metadata),
        grid=Grid.load(artifacts / "grid", len(metadata["data"]["features"])),
    )
