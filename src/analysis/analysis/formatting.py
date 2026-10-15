"""Polish number formatting: decimal comma and a true minus sign.

Matplotlib writes a decimal point. The formatters here wrap the default tick formatters and
replace it with a comma, independently of the system locale. In mathtext the comma is written
as `{,}`, otherwise it would be followed by a space like a punctuation mark.
"""

import math

import numpy as np
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from matplotlib.ticker import Formatter, LogFormatterSciNotation, ScalarFormatter

# Log-axis labels in this range are plain numbers (0.2, 1, 500), outside it powers of ten.
PLAIN_LOG_RANGE = (1e-3, 1e4)
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


# Characters of plain-text numbers (summary tables), by code point to keep the code ASCII.
DASH = chr(0x2014)
INFINITY = chr(0x221E)
THIN_SPACE = chr(0x202F)  # narrow no-break space: the Polish thousands separator
TIMES = chr(0x00B7)
_SUPERSCRIPTS = str.maketrans(
    "0123456789-",
    "".join(map(chr, [0x2070, 0xB9, 0xB2, 0xB3, 0x2074, 0x2075, 0x2076, 0x2077, 0x2078, 0x2079]))
    + chr(0x207B),
)
# Plain-text values in this range are written without a power of ten.
PLAIN_RANGE = (1e-3, 1e6)


def format_value(value: float | None, digits: int = 4) -> str:
    """Format a metric for plain text: `0,1303`, `22 910,7`, `7,325 * 10^-5` with superscripts.

    `digits` significant digits below 1000 (trailing zeros dropped), one decimal from 1000 up;
    `None` and NaN become a dash, infinities the infinity sign.
    """
    if value is None or math.isnan(value):
        return DASH
    if math.isinf(value):
        return (MINUS if value < 0 else "") + INFINITY
    magnitude = abs(value)
    low, high = PLAIN_RANGE
    if magnitude == 0:
        text = "0"
    elif magnitude < low or magnitude >= high:
        exponent = math.floor(math.log10(magnitude))
        mantissa = decimal_comma(f"{value / 10**exponent:.{digits - 1}f}").rstrip("0").rstrip(",")
        power = str(exponent).translate(_SUPERSCRIPTS)
        text = f"{mantissa}{TIMES}10{power}"
    elif magnitude >= 1000:
        text = decimal_comma(f"{value:,.1f}".replace(",", THIN_SPACE))
    else:
        decimals = max(digits - 1 - math.floor(math.log10(magnitude)), 0)
        text = decimal_comma(f"{value:.{decimals}f}")
        if "," in text:
            text = text.rstrip("0").rstrip(",")
    return text.replace("-", MINUS)


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


class PlainLogFormatter(LogFormatterSciNotation):
    """Log-axis labels as plain numbers in `PLAIN_LOG_RANGE`, powers of ten outside it.

    Labels the same ticks as the default formatter, but writes `0.2` instead of
    `2 x 10^-1`, which reads badly when an axis spans about one decade.
    """

    def __call__(self, x: float, pos: int | None = None) -> str:
        if not math.isfinite(x) or not super().__call__(x, pos):
            return ""
        return f"$\\mathdefault{{{plain_or_power(x)}}}$"


def plain_or_power(x: float) -> str:
    """Mathtext body of a log-axis label: `0.2`, `10^{-5}` or `3\\times10^{5}`."""
    low, high = PLAIN_LOG_RANGE
    if low <= abs(x) < high:
        return np.format_float_positional(float(f"{x:.3g}"), trim="-")
    exponent = math.floor(math.log10(abs(x)))
    coefficient = float(f"{x / 10**exponent:.3g}")
    if coefficient == 1:
        return f"10^{{{exponent}}}"
    return f"{coefficient:g}\\times10^{{{exponent}}}"


def use_decimal_comma(figure: Figure) -> None:
    """Wrap the tick formatters of every axis of the figure (call after setting the scales).

    Default log-axis formatters are replaced by `PlainLogFormatter`. Colorbars reset their
    formatter when drawn; create them with `format=DecimalCommaFormatter()` instead.
    """
    for axes in figure.axes:
        for axis in (axes.xaxis, axes.yaxis):
            for formatter, setter in (
                (axis.get_major_formatter(), axis.set_major_formatter),
                (axis.get_minor_formatter(), axis.set_minor_formatter),
            ):
                if isinstance(formatter, DecimalCommaFormatter):
                    continue
                if type(formatter) is LogFormatterSciNotation:
                    formatter = PlainLogFormatter(
                        labelOnlyBase=formatter.labelOnlyBase,
                        minor_thresholds=formatter.minor_thresholds,
                    )
                setter(DecimalCommaFormatter(formatter))
