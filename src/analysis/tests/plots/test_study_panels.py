import matplotlib.pyplot as plt

from analysis.plots.study_panels import category_ticks, fits
from analysis.style import new_figure


def test_fits_with_margin() -> None:
    assert fits(["abcd"], slot_cm=1.0, character_cm=0.2)  # 0,8 cm <= 0,9 cm
    assert not fits(["abcde"], slot_cm=1.0, character_cm=0.2)  # 1,0 cm > 0,9 cm
    assert fits([], slot_cm=0.0, character_cm=0.2)


def test_long_tick_labels_are_slanted() -> None:
    figure, axes = new_figure(16, 8)
    names = ["0", "16-16-16-16", "64-64-64-64"] * 3

    category_ticks(axes, names)

    labels = axes.get_xticklabels()
    assert [label.get_text() for label in labels] == names
    assert all(label.get_rotation() == 30 for label in labels)
    assert all(label.get_horizontalalignment() == "right" for label in labels)
    plt.close(figure)


def test_short_tick_labels_stay_horizontal() -> None:
    figure, axes = new_figure(16, 8)

    category_ticks(axes, ["sigmoid", "tanh", "relu", "leaky-relu"])

    assert all(label.get_rotation() == 0 for label in axes.get_xticklabels())
    plt.close(figure)
