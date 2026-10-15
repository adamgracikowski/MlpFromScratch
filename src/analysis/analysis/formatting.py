"""Polish number formatting: decimal comma and a true minus sign.

Matplotlib writes a decimal point. The formatters here wrap the default tick formatters and
replace it with a comma, independently of the system locale. In mathtext the comma is written
as `{,}`, otherwise it would be followed by a space like a punctuation mark.
"""

from matplotlib.axis import Axis
from matplotlib.figure import Figure
from matplotlib.ticker import Formatter, ScalarFormatter

MINUS = "\u2212"


def decimal_comma(text: str) -> str:
    """Replace the decimal point in a formatted number (plain text or mathtext)."""
    if text.startswith("$"):
        return text.replace(".", "{,}")
    return text.replace(".", ",")


def format_number(value: float, decimals: int) -> str:
    """Format a number for plot text: fixed decimals, decimal comma, true minus sign."""
    return decimal_comma(f"{value:.{decimals}f}").replace("-", MINUS)


def format_percent(fraction: float, decimals: int = 1) -> str:
    """Format a fraction as a percentage for plot text, e.g. `0.9406 -> 94,1%`."""
    return f"{format_number(100 * fraction, decimals)}%"


class DecimalCommaFormatter(Formatter):
    """Tick formatter delegating to another one and writing a decimal comma."""

    def __init__(self, inner: Formatter | None = None) -> None:
        self.inner = inner if inner is not None else ScalarFormatter()

    def set_axis(self, axis: Axis | None) -> None:
        super().set_axis(axis)
        self.inner.set_axis(axis)

    def set_locs(self, locs) -> None:
        self.inner.set_locs(locs)

    def __call__(self, x: float, pos: int | None = None) -> str:
        return decimal_comma(self.inner(x, pos))

    def format_ticks(self, values) -> list[str]:
        return [decimal_comma(text) for text in self.inner.format_ticks(values)]

    def get_offset(self) -> str:
        return decimal_comma(self.inner.get_offset())

    def format_data(self, value: float) -> str:
        return decimal_comma(self.inner.format_data(value))

    def format_data_short(self, value: float) -> str:
        return decimal_comma(self.inner.format_data_short(value))


def use_decimal_comma(figure: Figure) -> None:
    """Wrap the tick formatters of every axis of the figure (call after setting the scales).

    Colorbars reset their formatter when drawn; create them with
    `format=DecimalCommaFormatter()` instead.
    """
    for axes in figure.axes:
        for axis in (axes.xaxis, axes.yaxis):
            for formatter, setter in (
                (axis.get_major_formatter(), axis.set_major_formatter),
                (axis.get_minor_formatter(), axis.set_minor_formatter),
            ):
                if not isinstance(formatter, DecimalCommaFormatter):
                    setter(DecimalCommaFormatter(formatter))
