import json

import numpy as np
import polars as pl
import pytest

from analysis.experiment import ArtifactsError, SplitPredictions


def load(experiment_dir) -> dict[str, SplitPredictions]:
    artifacts = experiment_dir / "artifacts"
    metadata = json.loads((artifacts / "metadata.json").read_text(encoding="utf-8"))
    return SplitPredictions.load_all(artifacts / "predictions", metadata)


def test_loads_classification(make_experiment) -> None:
    predictions = load(make_experiment())

    assert list(predictions) == ["train", "test"]
    test = predictions["test"]
    assert test.inputs.shape == (30, 2)
    assert test.outputs.shape == (30, 3)
    assert test.labels.shape == test.predicted.shape == (30,)
    assert test.targets is None


def test_loads_regression(make_experiment) -> None:
    test = load(make_experiment(task="regression", features=1))["test"]

    assert test.inputs.shape == (30, 1)
    assert test.outputs.shape == (30, 1)
    assert test.targets.shape == (30,)
    assert test.labels is None and test.predicted is None


def test_inputs_come_from_csv_rows(make_experiment) -> None:
    experiment_dir = make_experiment(validation=True)
    predictions = load(experiment_dir)
    metadata = json.loads((experiment_dir / "artifacts/metadata.json").read_text(encoding="utf-8"))
    train_csv = pl.read_csv(metadata["config"]["data"]["train"]).select("x", "y").to_numpy()

    assert list(predictions) == ["train", "validation", "test"]
    for split in ("train", "validation"):
        np.testing.assert_array_equal(predictions[split].inputs, train_csv[predictions[split].rows])
    both = np.concatenate([predictions[split].rows for split in ("train", "validation")])
    assert sorted(both) == list(range(len(train_csv)))


def test_three_features(make_experiment) -> None:
    assert load(make_experiment(features=3))["train"].inputs.shape[1] == 3


def test_missing_dataset(make_experiment, tmp_path) -> None:
    experiment_dir = make_experiment()
    metadata_path = experiment_dir / "artifacts/metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["config"]["data"]["test"] = (tmp_path / "missing.csv").as_posix()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ArtifactsError, match="repository root"):
        load(experiment_dir)


def test_rows_past_end_of_csv(make_experiment) -> None:
    experiment_dir = make_experiment()
    np.save(experiment_dir / "artifacts/predictions/test/rows.npy", np.array([0, 999], np.uint64))

    with pytest.raises(ArtifactsError, match="past the end"):
        load(experiment_dir)
