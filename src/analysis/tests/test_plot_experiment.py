from analysis.experiment import load_experiment
from analysis.plot_experiment import PLOTS_DIR, plot_experiment

ALWAYS = {"loss", "metrics", "weights_trajectories", "weight_norms", "weights_heatmap"}
CLASSIFICATION = ALWAYS | {"confusion_train", "confusion_test"}
CLASSIFICATION_2D = CLASSIFICATION | {"dataset", "decision_boundary"}
REGRESSION_1D = ALWAYS | {"dataset", "regression"}


# Fast settings for tests that do not check the formats.
FAST = {"formats": ["png"], "dpi": 40}


def expected(names: set[str], formats=("pdf", "png")) -> set[str]:
    """File names of the plots in every format, plus the summary."""
    return {f"{name}.{file_format}" for name in names for file_format in formats} | {"summary.md"}


def file_names(report) -> set[str]:
    return {path.name for path in report.files}


def test_classification(make_experiment) -> None:
    report = plot_experiment(load_experiment(make_experiment()))

    assert report.directory.name == PLOTS_DIR
    assert file_names(report) == expected(CLASSIFICATION_2D)
    assert report.skipped == []
    assert all(path.stat().st_size > 0 for path in report.files)


def test_regression_with_validation(make_experiment) -> None:
    experiment_dir = make_experiment(task="regression", features=1, validation=True)
    report = plot_experiment(load_experiment(experiment_dir), **FAST)

    assert file_names(report) == expected(REGRESSION_1D, formats=["png"])
    assert report.skipped == []


def test_classification_with_validation(make_experiment) -> None:
    report = plot_experiment(load_experiment(make_experiment(validation=True)), **FAST)

    assert file_names(report) == expected(
        CLASSIFICATION_2D | {"confusion_validation"}, formats=["png"]
    )


def test_skips_input_space_plots_for_three_features(make_experiment) -> None:
    report = plot_experiment(load_experiment(make_experiment(features=3)), **FAST)

    skipped = dict(report.skipped)
    assert set(skipped) == {"dataset", "decision_boundary"}
    assert "2 input feature" in skipped["dataset"]
    assert file_names(report) == expected(CLASSIFICATION, formats=["png"])


def test_diverged(make_experiment) -> None:
    report = plot_experiment(load_experiment(make_experiment(diverged=True)), **FAST)

    assert file_names(report) == expected(CLASSIFICATION_2D, formats=["png"])


def test_diverged_regression(make_experiment) -> None:
    experiment_dir = make_experiment(task="regression", features=1, diverged=True)
    report = plot_experiment(load_experiment(experiment_dir), **FAST)

    assert file_names(report) == expected(REGRESSION_1D, formats=["png"])


def test_formats(make_experiment) -> None:
    report = plot_experiment(load_experiment(make_experiment()), formats=["svg"])

    assert {path.suffix for path in report.files} == {".svg", ".md"}


def test_removes_old_plots(make_experiment) -> None:
    experiment_dir = make_experiment()
    stale = experiment_dir / PLOTS_DIR / "stale.png"
    stale.parent.mkdir()
    stale.write_bytes(b"old")

    plot_experiment(load_experiment(experiment_dir), **FAST)

    assert not stale.exists()


def test_files_are_reproducible(make_experiment) -> None:
    experiment = load_experiment(make_experiment())
    first = {path.name: path.read_bytes() for path in plot_experiment(experiment).files}
    second = {path.name: path.read_bytes() for path in plot_experiment(experiment).files}

    assert first == second
