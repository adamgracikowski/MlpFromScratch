import pytest

from analysis.cli import build_parser, main
from analysis.plot_experiment import DEFAULT_DPI, DEFAULT_FORMATS


def test_parses_plot() -> None:
    args = build_parser().parse_args(["plot", "data/experiments/regression/example-cube"])

    assert args.command == "plot"
    assert args.experiment_dir.as_posix() == "data/experiments/regression/example-cube"
    assert args.formats == list(DEFAULT_FORMATS)
    assert args.dpi == DEFAULT_DPI


def test_parses_options() -> None:
    args = build_parser().parse_args(["plot", "dir", "--format", "PNG, svg,png", "--dpi", "300"])

    assert args.formats == ["png", "svg"]
    assert args.dpi == 300


@pytest.mark.parametrize(
    "options",
    [[], ["plot"], ["plot", "dir", "--format", "jpg"], ["plot", "dir", "--dpi", "0"]],
)
def test_rejects_invalid_arguments(options: list[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(options)

    assert exit_info.value.code == 2


def test_missing_artifacts_exit_code(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["plot", str(tmp_path)])

    assert exit_info.value.code == 2
    assert "compute train" in capsys.readouterr().err


def test_plot_writes_files(make_experiment, capsys) -> None:
    experiment_dir = make_experiment(features=3)
    main(["plot", str(experiment_dir), "--format", "png", "--dpi", "40"])

    output = capsys.readouterr().out
    assert "skipped dataset" in output
    assert "plots:" in output
    assert (experiment_dir / "plots").is_dir()


def test_parses_aggregate() -> None:
    args = build_parser().parse_args(["aggregate", "data/experiments/classification/study"])

    assert args.command == "aggregate"
    assert args.study_dir.as_posix() == "data/experiments/classification/study"


def test_aggregate_reports_missing_runs(make_study, capsys) -> None:
    study_dir = make_study(missing=("runs/tanh__a/seed-2",))
    main(["aggregate", str(study_dir)])

    captured = capsys.readouterr()
    assert "runs/tanh__a/seed-2" in captured.err
    assert "runs:    11 of 12 trained" in captured.out
    assert (study_dir / "results" / "variants.csv").is_file()


def test_aggregate_without_index_exit_code(make_study, capsys) -> None:
    study_dir = make_study()
    (study_dir / "runs" / "index.csv").unlink()
    with pytest.raises(SystemExit) as exit_info:
        main(["aggregate", str(study_dir)])

    assert exit_info.value.code == 2
    assert "compute generate" in capsys.readouterr().err
