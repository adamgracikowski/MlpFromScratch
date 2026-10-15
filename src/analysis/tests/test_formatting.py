import matplotlib.pyplot as plt
import pytest
from matplotlib.ticker import ScalarFormatter

from analysis.formatting import (
    DASH,
    INFINITY,
    MINUS,
    THIN_SPACE,
    TIMES,
    DecimalCommaFormatter,
    PlainLogFormatter,
    decimal_comma,
    format_number,
    format_percent,
    format_value,
    plain_or_power,
    use_decimal_comma,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0.5", "0,5"),
        ("12", "12"),
        ("$\\mathdefault{-1.25}$", "$\\mathdefault{-1{,}25}$"),
        ("$\\mathdefault{10^{-3}}$", "$\\mathdefault{10^{-3}}$"),
    ],
)
def test_decimal_comma(text: str, expected: str) -> None:
    assert decimal_comma(text) == expected


def test_format_number() -> None:
    assert format_number(0.5, 2) == "0,50"
    assert format_number(-1.234, 1) == f"{MINUS}1,2"
    assert format_number(3, 0) == "3"


def test_format_percent() -> None:
    assert format_percent(0.94066) == "94,1%"
    assert format_percent(1.0, 0) == "100%"


def tick_labels(axis) -> list[str]:
    figure = axis.get_figure()
    figure.canvas.draw()
    return [label.get_text() for label in axis.get_ticklabels()]


def test_wraps_scalar_axis() -> None:
    with plt.rc_context({"axes.formatter.use_mathtext": True}):
        figure, axes = plt.subplots()
        axes.plot([0, 1], [-1.5, 1.5])
        use_decimal_comma(figure)

        labels = tick_labels(axes.yaxis)
    plt.close(figure)

    assert isinstance(axes.yaxis.get_major_formatter(), DecimalCommaFormatter)
    assert any("{,}5" in label for label in labels)
    assert not any("." in label for label in labels)


def test_wraps_offset() -> None:
    formatter = DecimalCommaFormatter(ScalarFormatter(useOffset=True))
    figure, axes = plt.subplots()
    axes.plot([0, 1], [100000.0, 100000.5])
    axes.yaxis.set_major_formatter(formatter)
    tick_labels(axes.yaxis)
    plt.close(figure)

    assert "." not in formatter.get_offset()


def test_replaces_log_formatter_once() -> None:
    figure, axes = plt.subplots()
    axes.set_yscale("log")
    axes.plot([0, 1], [1e-3, 1])
    use_decimal_comma(figure)
    use_decimal_comma(figure)
    plt.close(figure)

    formatter = axes.yaxis.get_major_formatter()
    assert isinstance(formatter.inner, PlainLogFormatter)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.2, "0.2"),
        (1.0, "1"),
        (500.0, "500"),
        (1e-5, "10^{-5}"),
        (3e5, r"3\times10^{5}"),
    ],
)
def testplain_or_power(value: float, expected: str) -> None:
    assert plain_or_power(value) == expected


def test_plain_log_axis_labels() -> None:
    with plt.rc_context({"axes.formatter.use_mathtext": True}):
        figure, axes = plt.subplots()
        axes.set_yscale("log")
        axes.set_ylim(0.15, 1.2)
        use_decimal_comma(figure)

        labels = tick_labels(axes.yaxis) + [
            label.get_text() for label in axes.yaxis.get_minorticklabels()
        ]
    plt.close(figure)

    assert r"$\mathdefault{1}$" in labels
    assert r"$\mathdefault{0{,}2}$" in labels
    assert not any("times" in label for label in labels)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.13031287, "0,1303"),
        (0.1, "0,1"),
        (94.93, "94,93"),
        (22910.6528, f"22{THIN_SPACE}910,7"),
        (7.325e-5, f"7,325{TIMES}10{chr(0x207B)}{chr(0x2075)}"),
        (1e-8, f"1{TIMES}10{chr(0x207B)}{chr(0x2078)}"),
        (0.0, "0"),
        (-1.5, f"{MINUS}1,5"),
        (float("inf"), INFINITY),
        (float("nan"), DASH),
        (None, DASH),
    ],
)
def test_format_value(value, expected: str) -> None:
    assert format_value(value) == expected
