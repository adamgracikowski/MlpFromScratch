import matplotlib.pyplot as plt
import pytest

from analysis import labels
from analysis.plots.study_seeds import plot_study_seeds
from analysis.study import load_results, load_study


@pytest.mark.parametrize(
    ("task", "entry"),
    [("classification", labels.MEAN_AND_SD), ("regression", labels.MEDIAN_AND_QUARTILES)],
)
def test_legend_names_the_spread(make_study, task, entry) -> None:
    results = load_results(load_study(make_study(task=task)))

    figure = plot_study_seeds(results)

    assert figure.legends[0].get_texts()[0].get_text() == entry
    plt.close(figure)
