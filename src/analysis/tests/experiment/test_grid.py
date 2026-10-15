from analysis.experiment import Grid


def test_loads_2d(make_experiment) -> None:
    grid = Grid.load(make_experiment() / "artifacts/grid", feature_count=2)

    assert len(grid.axes) == 2
    assert grid.outputs.shape == (5, 5, 3)


def test_loads_1d(make_experiment) -> None:
    experiment_dir = make_experiment(task="regression", features=1)
    grid = Grid.load(experiment_dir / "artifacts/grid", feature_count=1)

    assert len(grid.axes) == 1
    assert grid.outputs.shape == (5, 1)


def test_absent_for_three_features(make_experiment) -> None:
    assert Grid.load(make_experiment(features=3) / "artifacts/grid", feature_count=3) is None
