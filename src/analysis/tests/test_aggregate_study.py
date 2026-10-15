import polars as pl
import pytest

from analysis.aggregate_study import RESULTS_DIR, aggregate_study
from analysis.study import StudyError, load_study


def test_writes_tables(make_study) -> None:
    study_dir = make_study(missing=("runs/tanh__a/seed-2",))
    stale = study_dir / RESULTS_DIR / "stale.txt"
    stale.parent.mkdir()
    stale.write_text("old", encoding="utf-8")

    report = aggregate_study(load_study(study_dir))

    assert sorted(path.name for path in (study_dir / RESULTS_DIR).iterdir()) == [
        "runs.csv",
        "variants.csv",
    ]
    assert report.missing == ["runs/tanh__a/seed-2"]
    assert pl.read_csv(study_dir / RESULTS_DIR / "runs.csv").height == 12
    assert pl.read_csv(study_dir / RESULTS_DIR / "variants.csv").height == 4


def test_writes_identical_files_twice(make_study) -> None:
    study = load_study(make_study())
    first = {path.name: path.read_bytes() for path in aggregate_study(study).files}
    second = {path.name: path.read_bytes() for path in aggregate_study(study).files}

    assert first == second


def test_study_without_trained_runs_is_an_error(make_study) -> None:
    study_dir = make_study(axes={"a": ["x"]}, seeds=(1,), missing=("runs/x/seed-1",))

    with pytest.raises(StudyError, match="compute train"):
        aggregate_study(load_study(study_dir))
