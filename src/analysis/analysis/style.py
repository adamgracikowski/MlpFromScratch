"""Common look of all plots: fonts, sizes, colors and saving.

Figures are drawn at their final print size (the text width of an A4 report is 16 cm), so
they are inserted into the report without scaling and the 10 pt font matches the text.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from analysis.formatting import use_decimal_comma

CM = 1 / 2.54  # inches per centimeter
TEXT_WIDTH = 16 * CM

# Okabe & Ito (2008), a palette distinguishable with color vision deficiencies.
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"
SKY_BLUE = "#56B4E9"
YELLOW = "#F0E442"
BLACK = "#000000"
CATEGORICAL = [BLUE, VERMILLION, GREEN, ORANGE, PURPLE, SKY_BLUE, YELLOW, BLACK]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "p"]

SPLIT_COLORS = {"train": BLUE, "validation": GREEN, "test": VERMILLION}
INK = "#222222"  # text and axes
MUTED = "#6b6b6b"  # secondary lines
GRID = "#e3e3e3"

STYLE = {
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "axes.prop_cycle": cycler(color=CATEGORICAL),
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": True,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "figure.dpi": 100,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "image.interpolation": "nearest",
    "savefig.bbox": "tight",  # crops empty margins, keeps the scale
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,  # embed TrueType: text stays text in the PDF
}

# Without dates and tool versions the files are identical for identical artifacts.
_METADATA = {
    "pdf": {"CreationDate": None},
    "png": {"Software": None},
    "svg": {"Date": None},
}
FORMATS = tuple(_METADATA)


def class_style(index: int) -> tuple[str, str]:
    """Color and marker of the class with the given index (sorted class labels)."""
    return CATEGORICAL[index % len(CATEGORICAL)], MARKERS[index % len(MARKERS)]


def new_figure(
    width_cm: float, height_cm: float, rows: int = 1, columns: int = 1, **kwargs
) -> tuple[Figure, Axes]:
    """Create a figure of the given print size with constrained layout."""
    return plt.subplots(
        rows, columns, figsize=(width_cm * CM, height_cm * CM), layout="constrained", **kwargs
    )


def save_figure(
    figure: Figure, directory: Path, name: str, formats: list[str], dpi: int
) -> list[Path]:
    """Save the figure as `<directory>/<name>.<format>` for every format, then close it."""
    use_decimal_comma(figure)
    paths = []
    for file_format in formats:
        path = directory / f"{name}.{file_format}"
        figure.savefig(path, dpi=dpi, metadata=_METADATA[file_format])
        paths.append(path)
    plt.close(figure)
    return paths
