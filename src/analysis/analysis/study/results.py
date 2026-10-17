"""Results of all runs of a study and their aggregation over seeds."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from analysis.experiment import SPLITS, ArtifactsError
from analysis.study.range_errors import RANGE_COLUMNS, InputCache, range_errors
from analysis.study.run_summary import FINISHED, RunSummary
from analysis.study.study import Study

CLASSIFICATION_METRICS = ("loss", "accuracy")
REGRESSION_METRICS = ("loss", "mse", "mae")
MISSING = "missing"
STATISTICS = ("mean", "sd", "min", "q1", "median", "q3", "max")


def task_metrics(study: Study) -> tuple[str, ...]:
    return CLASSIFICATION_METRICS if study.is_classification else REGRESSION_METRICS


def primary_metric(study: Study) -> str:
    """The metric compared across variants: the loss is not comparable between losses."""
    return "accuracy" if study.is_classification else "mse"


@dataclass(frozen=True)
class StudyResults:
    """One row per run of the index (missing runs too) and the loaded run summaries."""

    study: Study
    runs: pl.DataFrame
    summaries: dict[str, RunSummary]  # by `dir` of the index; missing runs left out
    metric_columns: list[str]  # `<split>_<metric>`; validation only if any run has it, then
    # the test MSE inside and outside the training range (regression)

    @property
    def missing(self) -> list[str]:
        return self.runs.filter(pl.col("status") == MISSING)["dir"].to_list()


def load_results(study: Study) -> StudyResults:
    """Read the summary of every run in the index; runs without artifacts are `missing`.
    Regression runs also get the test MSE inside and outside their training range."""
    metrics = task_metrics(study)
    range_columns = [] if study.is_classification else list(RANGE_COLUMNS)
    inputs = InputCache()
    summaries: dict[str, RunSummary] = {}
    rows = []
    for row in study.index.iter_rows(named=True):
        try:
            summary = RunSummary.load(study.directory / row["dir"])
        except ArtifactsError:
            summary = None
        record = {
            **row,
            "status": MISSING,
            "epochs": None,
            "best_epoch": None,
            "training_seconds": None,
        }
        for split in SPLITS:
            for metric in metrics:
                record[f"{split}_{metric}"] = None
        record |= dict.fromkeys(range_columns)
        if summary is not None:
            summaries[row["dir"]] = summary
            result = summary.metadata["result"]
            record.update(
                status=summary.stop_reason,
                epochs=result["epochs"],
                best_epoch=result["best_epoch"],
                training_seconds=summary.metadata["training_seconds"],
            )
            for split in SPLITS:
                for metric in metrics:
                    record[f"{split}_{metric}"] = summary.metric(split, metric)
            if range_columns:
                record.update(range_errors(summary, inputs))
        rows.append(record)

    schema: dict[str, pl.DataType] = dict(study.index.schema)
    schema |= {
        "status": pl.String(),
        "epochs": pl.Int64(),
        "best_epoch": pl.Int64(),
        "training_seconds": pl.Float64(),
    }
    all_metrics = [f"{split}_{metric}" for split in SPLITS for metric in metrics]
    schema |= {column: pl.Float64() for column in [*all_metrics, *range_columns]}
    runs = pl.DataFrame(rows, schema=schema, orient="row")
    validation = any(
        summary.metadata["data"]["samples"]["validation"] is not None
        for summary in summaries.values()
    )
    metric_columns = [
        column for column in all_metrics if validation or not column.startswith("validation_")
    ] + range_columns
    runs = runs.drop([column for column in all_metrics if column not in metric_columns])
    return StudyResults(study, runs, summaries, metric_columns)


def variants_table(results: StudyResults) -> pl.DataFrame:
    """One row per variant: run counts and statistics of the metrics over seeds.

    Statistics use the finished runs only (completed or early stopped); diverged and
    missing runs are counted separately. `sd` is the sample standard deviation, `q1` and
    `q3` are the quartiles (linear interpolation, as `numpy.percentile`).
    """
    finished = pl.col("status").is_in(FINISHED)
    expressions = [
        pl.len().cast(pl.Int64).alias("runs"),
        finished.sum().cast(pl.Int64).alias("finished"),
        (pl.col("status") == "diverged").sum().cast(pl.Int64).alias("diverged"),
        (pl.col("status") == MISSING).sum().cast(pl.Int64).alias("missing"),
        pl.col("epochs").filter(finished).mean().alias("epochs_mean"),
        pl.col("training_seconds").filter(finished).mean().alias("training_seconds_mean"),
    ]
    for column in results.metric_columns:
        values = pl.col(column).filter(finished)
        expressions += [
            values.mean().alias(f"{column}_mean"),
            values.std(ddof=1).alias(f"{column}_sd"),
            values.min().alias(f"{column}_min"),
            values.quantile(0.25, interpolation="linear").alias(f"{column}_q1"),
            values.median().alias(f"{column}_median"),
            values.quantile(0.75, interpolation="linear").alias(f"{column}_q3"),
            values.max().alias(f"{column}_max"),
        ]
    keys = ["variant", *results.study.axis_names]
    return results.runs.group_by(keys, maintain_order=True).agg(expressions)


def representative_runs(results: StudyResults) -> dict[str, Path]:
    """Per variant, the finished run whose primary test metric is closest to the median
    over seeds (the smaller seed on a tie); variants without finished runs are left out."""
    column = f"test_{primary_metric(results.study)}"
    chosen = {}
    finished = results.runs.filter(pl.col("status").is_in(FINISHED) & pl.col(column).is_not_null())
    for (variant,), runs in finished.group_by("variant", maintain_order=True):
        values = runs[column].to_numpy()
        distance = np.abs(values - np.median(values))
        # Equal distances on both sides of an even-count median differ by rounding.
        closest = np.flatnonzero(np.isclose(distance, distance.min(), rtol=1e-9, atol=1e-12))
        seeds = runs["seed"].to_numpy()[closest]
        chosen[variant] = results.study.directory / runs["dir"][int(closest[seeds.argmin()])]
    return chosen
