import matplotlib.pyplot as plt

from analysis.plots.study_heatmap import PLUS_MINUS, plot_study_heatmap
from analysis.study import load_results, load_study

REPORT = '[report]\nx = "a"\ncolor = "b"'


def _cell_texts(figure) -> list[str]:
    return [text.get_text() for text in figure.axes[0].texts]


def test_cells_on_one_line_when_they_fit(make_study) -> None:
    study = make_study(task="regression", axes={"a": ["x", "y"], "b": ["1"]}, report=REPORT)

    figure = plot_study_heatmap(load_results(load_study(study)))

    texts = _cell_texts(figure)
    assert len(texts) == 2
    assert all(f" {PLUS_MINUS} " in text and "\n" not in text for text in texts)
    plt.close(figure)


def test_cells_on_two_lines_when_too_narrow(make_study) -> None:
    levels = [f"level{index}" for index in range(12)]
    study = make_study(task="regression", axes={"a": levels, "b": ["1"]}, report=REPORT)

    figure = plot_study_heatmap(load_results(load_study(study)))

    assert all(f"\n{PLUS_MINUS} " in text for text in _cell_texts(figure))
    plt.close(figure)
