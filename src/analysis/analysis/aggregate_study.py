"""Results of a study aggregated over seeds, written to `<study>/results/`."""

import shutil
from dataclasses import dataclass
from pathlib import Path

from analysis.study import Study, StudyError, load_results, variants_table

RESULTS_DIR = "results"


@dataclass(frozen=True)
class AggregateReport:
    """Written files and the runs that have no artifacts yet."""

    directory: Path
    files: list[Path]
    missing: list[str]  # `dir` of the index, relative to the study directory


def aggregate_study(study: Study) -> AggregateReport:
    """Write `runs.csv` (one row per run) and `variants.csv` (statistics over seeds).

    `results/` is removed first. Runs without artifacts are reported as missing; a study
    without any trained run is an error.
    """
    results = load_results(study)
    if not results.summaries:
        raise StudyError(
            f"no run of {study.directory.as_posix()} has artifacts: "
            f"run `compute train {study.directory.as_posix()}` first"
        )
    directory = study.directory / RESULTS_DIR
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir()

    files = [directory / "runs.csv", directory / "variants.csv"]
    results.runs.write_csv(files[0])
    variants_table(results).write_csv(files[1])
    return AggregateReport(directory=directory, files=files, missing=results.missing)
