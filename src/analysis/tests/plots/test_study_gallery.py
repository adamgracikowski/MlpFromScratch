import matplotlib.pyplot as plt

from analysis import labels
from analysis.plots.study_gallery import gallery_skip_reason, plot_study_gallery
from analysis.study import load_results, load_study


def _titles(figure) -> list[str]:
    return [axes.get_title() for axes in figure.axes if axes.get_visible()]


def test_figure_per_level_of_the_other_axes(make_study) -> None:
    results = load_results(load_study(make_study()))  # x = dataset, by default the last axis

    figures = plot_study_gallery(results)

    assert [name for name, _ in figures] == ["gallery_sigmoid", "gallery_tanh"]
    _, figure = figures[0]
    titles = _titles(figure)
    assert [title.split(" ")[0] for title in titles] == ["a", "b"]
    assert all(title.endswith("%") for title in titles)
    assert [text.get_text() for text in figure.legends[0].get_texts()][-1] == labels.MISCLASSIFIED
    for _, figure in figures:
        plt.close(figure)


def test_regression_panels_show_the_mse(make_study) -> None:
    results = load_results(
        load_study(make_study(task="regression", features=1, axes={"a": ["x", "y"]}))
    )

    ((name, figure),) = plot_study_gallery(results)

    assert name == "gallery"
    assert [title.split(" ")[:3] for title in _titles(figure)] == [
        ["x", chr(0xB7), "MSE"],
        ["y", chr(0xB7), "MSE"],
    ]
    plt.close(figure)


def test_four_panels_in_two_columns(make_study) -> None:
    results = load_results(load_study(make_study(axes={"a": ["w", "x", "y", "z"]}, seeds=(1,))))

    ((_, figure),) = plot_study_gallery(results)

    visible = [axes for axes in figure.axes if axes.get_visible()]
    assert len(visible) == 4
    assert visible[0].get_gridspec().get_geometry() == (2, 2)
    plt.close(figure)


def test_variant_without_finished_run(make_study) -> None:
    missing = tuple(f"runs/tanh__a/seed-{seed}" for seed in (1, 2, 3))
    results = load_results(load_study(make_study(missing=missing)))

    figures = dict(plot_study_gallery(results))

    titles = _titles(figures["gallery_tanh"])
    assert titles[0] == labels.gallery_title("a", labels.NO_FINISHED_RUN)
    assert titles[1].startswith("b ")
    for figure in figures.values():
        plt.close(figure)


def test_skip_reasons(make_study) -> None:
    trained = load_results(load_study(make_study()))
    diverged = load_results(
        load_study(make_study(axes={"a": ["x"]}, seeds=(1,), diverged=("runs/x/seed-1",)))
    )
    regression = load_results(load_study(make_study(task="regression", axes={"a": ["x"]})))

    assert gallery_skip_reason(trained) is None
    assert gallery_skip_reason(regression) == "needs 1 input feature(s), the data has 2"
    assert gallery_skip_reason(diverged) == "no variant has a finished run"
