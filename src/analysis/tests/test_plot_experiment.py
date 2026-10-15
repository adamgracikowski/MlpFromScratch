from analysis.experiment import load_experiment
from analysis.plot_experiment import PLOTS_DIR, plot_experiment


def file_names(report) -> set[str]:
    return {path.name for path in report.files}


def test_classification(make_experiment) -> None:
    report = plot_experiment(load_experiment(make_experiment()))

    assert report.directory.name == PLOTS_DIR
    assert {"dataset.pdf", "dataset.png"} <= file_names(report)
    assert report.skipped == []
    assert all(path.stat().st_size > 0 for path in report.files)


def test_regression_with_validation(make_experiment) -> None:
    experiment_dir = make_experiment(task="regression", features=1, validation=True)
    report = plot_experiment(load_experiment(experiment_dir))

    assert {"dataset.pdf", "dataset.png"} <= file_names(report)
    assert report.skipped == []


def test_classification_with_validation(make_experiment) -> None:
    report = plot_experiment(load_experiment(make_experiment(validation=True)), formats=["png"])

    assert "dataset.png" in file_names(report)


def test_skips_input_space_plots_for_three_features(make_experiment) -> None:
    report = plot_experiment(load_experiment(make_experiment(features=3)))

    skipped = dict(report.skipped)
    assert "dataset" in skipped
    assert "2 input feature" in skipped["dataset"]
    assert not any(name.startswith("dataset.") for name in file_names(report))


def test_formats(make_experiment) -> None:
    report = plot_experiment(load_experiment(make_experiment()), formats=["svg"])

    assert {path.suffix for path in report.files} == {".svg"}


def test_removes_old_plots(make_experiment) -> None:
    experiment_dir = make_experiment()
    stale = experiment_dir / PLOTS_DIR / "stale.png"
    stale.parent.mkdir()
    stale.write_bytes(b"old")

    plot_experiment(load_experiment(experiment_dir))

    assert not stale.exists()


def test_files_are_reproducible(make_experiment) -> None:
    experiment = load_experiment(make_experiment())
    first = {path.name: path.read_bytes() for path in plot_experiment(experiment).files}
    second = {path.name: path.read_bytes() for path in plot_experiment(experiment).files}

    assert first == second
