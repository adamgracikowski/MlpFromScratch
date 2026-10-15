import pytest

from analysis.cli import build_parser, main


def test_parses_plot() -> None:
    args = build_parser().parse_args(["plot", "data/experiments/regression/example-cube"])

    assert args.command == "plot"
    assert args.experiment_dir.as_posix() == "data/experiments/regression/example-cube"


def test_requires_command() -> None:
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args([])

    assert exit_info.value.code == 2


def test_missing_artifacts_exit_code(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["plot", str(tmp_path)])

    assert exit_info.value.code == 2
    assert "compute train" in capsys.readouterr().err


def test_plot_loads_experiment(make_experiment, capsys) -> None:
    main(["plot", str(make_experiment())])

    assert "classification" in capsys.readouterr().out
