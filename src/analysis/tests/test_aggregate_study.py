import polars as pl
import pytest

from analysis.aggregate_study import RESULTS_DIR, aggregate_study
from analysis.study import StudyError, load_study

PNG = {"formats": ["png"], "dpi": 40}


def _names(study_dir) -> list[str]:
    return sorted(path.name for path in (study_dir / RESULTS_DIR).iterdir())


def test_writes_tables_plots_and_summary(make_study) -> None:
    study_dir = make_study(missing=("runs/tanh__a/seed-2",))
    stale = study_dir / RESULTS_DIR / "stale.txt"
    stale.parent.mkdir()
    stale.write_text("old", encoding="utf-8")

    report = aggregate_study(load_study(study_dir), **PNG)

    assert _names(study_dir) == [
        "curves.png",
        "curves_train_loss.png",
        "gallery_sigmoid.png",
        "gallery_tanh.png",
        "runs.csv",
        "seeds.png",
        "summary.md",
        "variants.csv",
    ]
    assert [name for name, _ in report.skipped] == ["heatmap"]
    assert report.missing == ["runs/tanh__a/seed-2"]
    assert report.plotted_runs == {}
    assert pl.read_csv(study_dir / RESULTS_DIR / "runs.csv").height == 12
    assert pl.read_csv(study_dir / RESULTS_DIR / "variants.csv").height == 4


def test_heatmap_with_x_and_color(make_study) -> None:
    study_dir = make_study(report='[report]\nx = "dataset"\ncolor = "activation"')
    report = aggregate_study(load_study(study_dir), **PNG)

    assert report.skipped == []
    assert "heatmap.png" in _names(study_dir)


def test_skips_train_loss_of_different_losses(make_study) -> None:
    study_dir = make_study(losses={"tanh": "mse"})
    report = aggregate_study(load_study(study_dir), **PNG)

    reasons = dict(report.skipped)
    assert "different losses" in reasons["curves_train_loss"]


def test_regression_with_diverged_runs(make_study) -> None:
    study_dir = make_study(
        task="regression",
        features=1,
        report='[report]\nx = "activation"\ncolor = "dataset"',
        diverged=("runs/sigmoid__a/seed-1", "runs/sigmoid__a/seed-2", "runs/sigmoid__a/seed-3"),
    )
    report = aggregate_study(load_study(study_dir), **PNG)

    assert report.skipped == []
    summary = (study_dir / RESULTS_DIR / "summary.md").read_text(encoding="utf-8")
    assert "| `sigmoid` | `a` | 0/3 | 3 |" in summary


def test_study_without_axes(make_study) -> None:
    study_dir = make_study(axes={})
    report = aggregate_study(load_study(study_dir), **PNG)

    assert [name for name, _ in report.skipped] == ["heatmap"]
    assert "seeds.png" in _names(study_dir)


def test_plots_representative_runs(make_study) -> None:
    study_dir = make_study(axes={"activation": ["sigmoid", "tanh"]})
    report = aggregate_study(load_study(study_dir), plot_runs=True, **PNG)

    assert report.plotted_runs == {
        "sigmoid": study_dir / "runs/sigmoid/seed-2",
        "tanh": study_dir / "runs/tanh/seed-2",
    }
    assert (study_dir / "runs/tanh/seed-2/plots/loss.png").is_file()
    assert not (study_dir / "runs/tanh/seed-1/plots").exists()
    summary = (study_dir / RESULTS_DIR / "summary.md").read_text(encoding="utf-8")
    assert "`runs/tanh/seed-2`" in summary


def test_writes_identical_files_twice(make_study) -> None:
    study = load_study(make_study(report='[report]\nx = "dataset"\ncolor = "activation"'))
    first = {path.name: path.read_bytes() for path in aggregate_study(study, **PNG).files}
    second = {path.name: path.read_bytes() for path in aggregate_study(study, **PNG).files}

    assert first == second


def test_study_without_trained_runs_is_an_error(make_study) -> None:
    study_dir = make_study(axes={"a": ["x"]}, seeds=(1,), missing=("runs/x/seed-1",))

    with pytest.raises(StudyError, match="compute train"):
        aggregate_study(load_study(study_dir))
