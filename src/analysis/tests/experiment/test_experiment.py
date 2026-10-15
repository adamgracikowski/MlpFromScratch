import pytest

from analysis.experiment import ArtifactsError, load_experiment


def test_loads_classification(make_experiment) -> None:
    experiment = load_experiment(make_experiment())

    assert experiment.task == "classification"
    assert experiment.is_classification
    assert experiment.features == ["x", "y"]
    assert experiment.classes == [1, 2, 3]
    assert experiment.stop_reason == "completed"
    assert experiment.best_epoch is None
    assert list(experiment.predictions) == ["train", "test"]
    assert experiment.weights.layer_count == 2
    assert experiment.grid is not None
    assert experiment.epochs.columns == [
        "epoch",
        "iteration",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
    ]


def test_loads_regression(make_experiment) -> None:
    experiment = load_experiment(make_experiment(task="regression", features=1))

    assert not experiment.is_classification
    assert experiment.classes is None
    assert "train_mse" in experiment.epochs.columns


def test_early_stopped(make_experiment) -> None:
    experiment = load_experiment(make_experiment(validation=True, early_stopped=True))

    assert experiment.stop_reason == "early_stopped"
    assert experiment.best_epoch == 2
    assert "validation_loss" in experiment.epochs.columns


def test_missing_artifacts(tmp_path) -> None:
    with pytest.raises(ArtifactsError, match="compute train"):
        load_experiment(tmp_path)
