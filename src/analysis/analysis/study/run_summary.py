"""The results of one run of a study: metadata and epoch metrics, without arrays."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from analysis.experiment import ARTIFACTS_DIR, ArtifactsError

FINISHED = ("completed", "early_stopped")


@dataclass(frozen=True)
class RunSummary:
    """`metadata.json` and `epochs.csv` of one run; enough to compare runs."""

    directory: Path
    metadata: dict[str, Any]
    epochs: pl.DataFrame

    @classmethod
    def load(cls, run_dir: Path) -> "RunSummary":
        """Read `<run_dir>/artifacts/metadata.json` and `epochs.csv`."""
        artifacts = run_dir / ARTIFACTS_DIR
        metadata_path = artifacts / "metadata.json"
        if not metadata_path.is_file():
            raise ArtifactsError(f"no artifacts in {artifacts.as_posix()}")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            epochs = pl.read_csv(artifacts / "epochs.csv")
        except (OSError, ValueError, pl.exceptions.PolarsError) as error:
            raise ArtifactsError(f"cannot read {artifacts.as_posix()}: {error}") from error
        return cls(directory=run_dir, metadata=metadata, epochs=epochs)

    @property
    def stop_reason(self) -> str:
        return self.metadata["result"]["stop"]["reason"]

    @property
    def finished(self) -> bool:
        """Whether the run ended normally (completed or early stopped, not diverged)."""
        return self.stop_reason in FINISHED

    def metric(self, split: str, name: str) -> float | None:
        """Final metric of the returned network; `None` for a missing split or value."""
        metrics = self.metadata["result"]["metrics"].get(split)
        return None if metrics is None else metrics.get(name)
