import matplotlib.pyplot as plt

from analysis.plots.study_heatmap import EN_DASH, PLUS_MINUS, plot_study_heatmap
from analysis.study import load_results, load_study

REPORT = '[report]\nx = "a"\ncolor = "b"'


def _cell_texts(figure) -> list[str]:
    return [text.get_text() for text in figure.axes[0].texts]


def test_errors_as_median_and_quartiles_on_one_line(make_study) -> None:
    study = make_study(task="regression", axes={"a": ["x", "y"], "b": ["1"]}, report=REPORT)

    figure = plot_study_heatmap(load_results(load_study(study)))

    texts = _cell_texts(figure)
    assert len(texts) == 2
    assert all(" (" in text and EN_DASH in text and "\n" not in text for text in texts)
    plt.close(figure)


def test_accuracy_as_mean_and_deviation(make_study) -> None:
    study = make_study(axes={"a": ["x", "y"], "b": ["1"]}, report=REPORT)

    figure = plot_study_heatmap(load_results(load_study(study)))

    assert all(f" {PLUS_MINUS} " in text for text in _cell_texts(figure))
    plt.close(figure)


def test_cells_on_two_lines_when_too_narrow(make_study) -> None:
    levels = [f"level{index}" for index in range(12)]
    study = make_study(task="regression", axes={"a": levels, "b": ["1"]}, report=REPORT)

    figure = plot_study_heatmap(load_results(load_study(study)))

    assert all("\n" in text and "(" not in text for text in _cell_texts(figure))
    plt.close(figure)


def test_quartiles_on_own_lines_when_two_lines_too_wide(make_study) -> None:
    levels = [f"level{index}" for index in range(30)]
    study = make_study(task="regression", axes={"a": levels, "b": ["1"]}, report=REPORT)

    figure = plot_study_heatmap(load_results(load_study(study)))

    for text in _cell_texts(figure):
        lines = text.splitlines()
        assert len(lines) == 3 and lines[1].endswith(EN_DASH) and EN_DASH not in lines[2]
    plt.close(figure)


def test_labels_on_axes_and_ticks(make_study) -> None:
    report = REPORT + '\nlabels = { a = "Czynnik", b = "Rozmiar", x = "iks" }'
    study = make_study(axes={"a": ["x", "y"], "b": ["1"]}, report=report)

    figure = plot_study_heatmap(load_results(load_study(study)))

    axes = figure.axes[0]
    assert (axes.get_xlabel(), axes.get_ylabel()) == ("Czynnik", "Rozmiar")
    assert [tick.get_text() for tick in axes.get_xticklabels()] == ["iks", "y"]
    plt.close(figure)
