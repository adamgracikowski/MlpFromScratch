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
    main(["plot", str(experiment_dir), "--format", "png"])

    output = capsys.readouterr().out
    assert "skipped dataset" in output
    assert "plots:" in output
    assert (experiment_dir / "plots").is_dir()
