"""`summary.md` of a study: plan, base config and results aggregated over seeds."""

from pathlib import Path
from typing import Any

import polars as pl

from analysis import labels
from analysis.formatting import DASH, format_number, format_percent, format_value
from analysis.study import StudyResults
from analysis.summary import config_value, markdown_table, tagged

SUMMARY_FILE = "summary.md"
PLUS_MINUS = chr(0xB1)


def write_study_summary(
    results: StudyResults,
    variants: pl.DataFrame,
    representative: dict[str, Path],
    directory: Path,
) -> Path:
    """Write `<directory>/summary.md` and return its path."""
    path = directory / SUMMARY_FILE
    text = study_summary_markdown(results, variants, representative)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def study_summary_markdown(
    results: StudyResults, variants: pl.DataFrame, representative: dict[str, Path]
) -> str:
    """The summary of a study as Markdown (Polish, like the report). Representative runs
    are listed when their plots were drawn."""
    study = results.study
    sections = [f"# {labels.STUDY_TITLE} `{study.directory.name}`"]
    if study.description:
        sections.append(study.description)
    sections += [
        f"## {labels.SECTION_PLAN}",
        markdown_table(
            [labels.AXIS, labels.KEY, labels.LEVELS],
            [
                [
                    f"`{axis.name}`",
                    ", ".join(f"`{key}`" for key in axis.keys),
                    ", ".join(f"`{level}`" for level in axis.levels),
                ]
                for axis in study.axes
            ],
        ),
        markdown_table(
            [labels.PARAMETER, labels.VALUE],
            [
                [labels.SEEDS, ", ".join(str(seed) for seed in study.seeds)],
                [
                    labels.RUN_COUNT,
                    f"{study.index.height} ({len(study.variants)} x {len(study.seeds)})",
                ],
            ],
        ),
        f"## {labels.SECTION_BASE}",
        markdown_table([labels.PARAMETER, labels.VALUE], _base_rows(results)),
        f"## {labels.SECTION_RESULTS}",
        markdown_table(*_result_rows(results, variants)),
        labels.RESULTS_NOTE
        + ("" if study.is_classification else " " + labels.REGRESSION_METRICS_NOTE),
    ]
    if representative:
        rows = [
            [f"`{variant}`", f"`{run.relative_to(study.directory).as_posix()}`"]
            for variant, run in representative.items()
        ]
        sections += [
            f"## {labels.SECTION_REPRESENTATIVE}",
            markdown_table([labels.VARIANT, labels.RUN], rows),
            labels.REPRESENTATIVE_NOTE,
        ]
    return "\n\n".join(sections) + "\n"


def _base_rows(results: StudyResults) -> list[list[str]]:
    """Every key of `[base]` with its value; keys set by an axis name that axis."""
    set_by = {key: axis.name for axis in results.study.axes for key in axis.keys}
    rows = []
    for key, value in _flatten(results.study.base):
        overriding = next(
            (
                axis
                for prefix, axis in set_by.items()
                if key == prefix or key.startswith(prefix + ".")
            ),
            None,
        )
        text = labels.set_by_axis(overriding) if overriding else _value(value)
        rows.append([f"`{key}`", text])
    return rows


def _flatten(table: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    """Dotted keys and values; tables with a `kind` tag are values, not sections."""
    items = []
    for name, value in table.items():
        key = f"{prefix}{name}"
        if isinstance(value, dict) and "kind" not in value:
            items += _flatten(value, f"{key}.")
        else:
            items.append((key, value))
    return items


def _value(value: Any) -> str:
    if isinstance(value, bool):
        return f"`{str(value).lower()}`"
    if isinstance(value, dict):
        return tagged(value)
    if isinstance(value, list):
        return "`[" + ", ".join(str(item) for item in value) + "]`"
    return config_value(value)


def _result_rows(
    results: StudyResults, variants: pl.DataFrame
) -> tuple[list[str], list[list[str]]]:
    study = results.study
    splits = [
        split
        for split in ("train", "validation", "test")
        if f"{split}_loss" in results.metric_columns
    ]
    if study.is_classification:
        metrics = [(labels.ACCURACY_COLUMN, f"{split}_accuracy", split) for split in splits]
    else:
        metrics = [(labels.MSE_COLUMN, f"{split}_mse", split) for split in splits]
        metrics.append((labels.MAE_COLUMN, "test_mae", "test"))
    missing = int(variants["missing"].sum()) > 0
    header = [
        *(f"`{name}`" for name in study.axis_names or [labels.VARIANT]),
        labels.FINISHED,
        labels.DIVERGED,
        *([labels.MISSING] if missing else []),
        *(labels.metric_column(name, split) for name, _, split in metrics),
        labels.MEAN_EPOCHS,
        labels.MEAN_TIME,
    ]
    rows = []
    for row in variants.iter_rows(named=True):
        cells = [f"`{row[name]}`" for name in study.axis_names] or [f"`{row['variant']}`"]
        cells += [f"{row['finished']}/{row['runs']}", str(row["diverged"])]
        if missing:
            cells.append(str(row["missing"]))
        for _, column, _ in metrics:
            cells.append(
                _mean_sd(row[f"{column}_mean"], row[f"{column}_sd"], study.is_classification)
            )
        epochs = row["epochs_mean"]
        cells.append(DASH if epochs is None else format_number(epochs, 1))
        time = row["training_seconds_mean"]
        cells.append(DASH if time is None else labels.seconds(format_value(time, 3)))
        rows.append(cells)
    return header, rows


def _mean_sd(mean: float | None, sd: float | None, percent: bool) -> str:
    """`97,20% ± 0,40` (accuracy, the deviation in percentage points) or `1234 ± 56`."""
    if mean is None:
        return DASH
    if percent:
        main = format_percent(mean, 2)
        spread = None if sd is None else format_number(100 * sd, 2)
    else:
        main = format_value(mean)
        spread = None if sd is None else format_value(sd, 2)
    return main if spread is None else f"{main} {PLUS_MINUS} {spread}"
