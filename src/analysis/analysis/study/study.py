"""A study written by `compute generate`: `study.toml` and `runs/index.csv`."""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

STUDY_FILE = "study.toml"
RUNS_DIR = "runs"
INDEX_FILE = "index.csv"
INDEX_COLUMNS = ("dir", "variant", "seed")


class StudyError(Exception):
    """A study directory is incomplete or cannot be read."""


@dataclass(frozen=True)
class Axis:
    """An axis of the study and its level names in `study.toml` order."""

    name: str
    levels: list[str]


@dataclass(frozen=True)
class Roles:
    """Placement of the axes on the comparison plots, by axis name."""

    x: str | None
    color: str | None
    panels: list[str]


@dataclass(frozen=True)
class Study:
    """Parameters of a study and the index of its runs (one row per run)."""

    directory: Path
    description: str | None
    seeds: list[int]
    axes: list[Axis]
    roles: Roles
    base: dict[str, Any]
    index: pl.DataFrame  # dir, variant, seed, then one column per axis

    @property
    def task(self) -> str:
        return self.base["task"]

    @property
    def is_classification(self) -> bool:
        return self.task == "classification"

    @property
    def axis_names(self) -> list[str]:
        return [axis.name for axis in self.axes]

    @property
    def variants(self) -> list[str]:
        """Variant names in generation order."""
        return list(dict.fromkeys(self.index["variant"].to_list()))


def load_study(study_dir: Path) -> Study:
    """Read `<study_dir>/study.toml` and `<study_dir>/runs/index.csv`."""
    study_path = study_dir / STUDY_FILE
    index_path = study_dir / RUNS_DIR / INDEX_FILE
    if not study_path.is_file():
        raise StudyError(f"no {STUDY_FILE} in {study_dir.as_posix()}")
    if not index_path.is_file():
        raise StudyError(
            f"no {index_path.as_posix()}: run `compute generate {study_dir.as_posix()}` first"
        )
    try:
        config = tomllib.loads(study_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise StudyError(f"cannot parse {study_path.as_posix()}: {error}") from error

    axis_names = [axis["name"] for axis in config.get("axes", [])]
    index = pl.read_csv(
        index_path,
        schema_overrides={name: pl.String for name in ("dir", "variant", *axis_names)},
    )
    expected = [*INDEX_COLUMNS, *axis_names]
    if index.columns != expected:
        raise StudyError(
            f"{index_path.as_posix()} has columns {index.columns}, expected {expected}: "
            f"run `compute generate {study_dir.as_posix()}` again"
        )
    axes = [Axis(name, list(dict.fromkeys(index[name].to_list()))) for name in axis_names]
    return Study(
        directory=study_dir,
        description=config.get("description"),
        seeds=list(config["seeds"]),
        axes=axes,
        roles=_roles(config.get("report", {}), axis_names),
        base=config["base"],
        index=index,
    )


def _roles(report: dict[str, Any], axis_names: list[str]) -> Roles:
    """Roles from `[report]`; axes without a role become panels. Without any role the
    last axis is on the x axis."""
    x = report.get("x")
    color = report.get("color")
    panels = list(report.get("panels", []))
    if x is None and color is None and not panels and axis_names:
        x = axis_names[-1]
    assigned = {x, color, *panels}
    panels += [name for name in axis_names if name not in assigned]
    return Roles(x=x, color=color, panels=panels)
