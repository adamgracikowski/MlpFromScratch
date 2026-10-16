import pytest

from analysis.study import load_results, load_study, representative_runs, variants_table


def test_runs_table_has_one_row_per_run(make_study) -> None:
    results = load_results(load_study(make_study()))

    assert results.runs.height == 12
    assert results.metric_columns == ["train_loss", "train_accuracy", "test_loss", "test_accuracy"]
    first = results.runs.row(0, named=True)
    assert first["dir"] == "runs/sigmoid__a/seed-1"
    assert first["status"] == "completed"
    assert first["epochs"] == 4
    assert first["test_accuracy"] == pytest.approx(0.901)
    assert results.missing == []


def test_regression_metrics(make_study) -> None:
    results = load_results(load_study(make_study(task="regression", axes={"a": ["x"]})))

    assert results.metric_columns == [
        "train_loss",
        "train_mse",
        "train_mae",
        "test_loss",
        "test_mse",
        "test_mae",
        "test_mse_inside",
        "test_mse_outside",
    ]
    assert results.runs["test_mse_inside"].null_count() == 0


def test_marks_missing_and_diverged_runs(make_study) -> None:
    study = load_study(
        make_study(missing=("runs/tanh__a/seed-2",), diverged=("runs/tanh__a/seed-3",))
    )
    results = load_results(study)

    statuses = dict(zip(results.runs["dir"], results.runs["status"], strict=True))
    assert statuses["runs/tanh__a/seed-2"] == "missing"
    assert statuses["runs/tanh__a/seed-3"] == "diverged"
    assert results.missing == ["runs/tanh__a/seed-2"]
    assert "runs/tanh__a/seed-2" not in results.summaries


def test_variants_aggregate_finished_runs_only(make_study) -> None:
    study = load_study(
        make_study(missing=("runs/tanh__a/seed-2",), diverged=("runs/tanh__a/seed-3",))
    )
    variants = variants_table(load_results(study))

    assert variants["variant"].to_list() == study.variants
    assert variants.columns[:7] == [
        "variant",
        "activation",
        "dataset",
        "runs",
        "finished",
        "diverged",
        "missing",
    ]
    full = variants.row(0, named=True)
    assert (full["runs"], full["finished"], full["diverged"], full["missing"]) == (3, 3, 0, 0)
    assert full["test_accuracy_mean"] == pytest.approx(0.902)
    assert full["test_accuracy_sd"] == pytest.approx(0.001)
    assert full["test_accuracy_min"] == pytest.approx(0.901)
    assert full["test_accuracy_median"] == pytest.approx(0.902)

    partial = variants.row(2, named=True)
    assert (partial["finished"], partial["diverged"], partial["missing"]) == (1, 1, 1)
    assert partial["test_accuracy_mean"] == pytest.approx(0.941)
    assert partial["test_accuracy_sd"] is None


def test_variant_without_finished_runs_has_no_statistics(make_study) -> None:
    study = load_study(make_study(axes={"a": ["x"]}, seeds=(1,), diverged=("runs/x/seed-1",)))
    row = variants_table(load_results(study)).row(0, named=True)

    assert (row["finished"], row["diverged"]) == (0, 1)
    assert row["test_accuracy_mean"] is None


def test_representative_run_is_closest_to_median(make_study) -> None:
    study_dir = make_study(seeds=(1, 2, 3, 4), diverged=("runs/tanh__b/seed-1",))
    runs = representative_runs(load_results(load_study(study_dir)))

    assert list(runs) == ["sigmoid__a", "sigmoid__b", "tanh__a", "tanh__b"]
    # Four seeds: the median falls between seeds 2 and 3, the smaller seed wins.
    assert runs["sigmoid__a"] == study_dir / "runs/sigmoid__a/seed-2"
    # Seeds 2-4 finished: seed 3 is the median.
    assert runs["tanh__b"] == study_dir / "runs/tanh__b/seed-3"
