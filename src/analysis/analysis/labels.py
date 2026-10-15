"""Polish texts shown on the plots (the report is written in Polish).

This is the only module with non-ASCII text: Polish words need their diacritics.
"""

SPLIT_NAMES = {
    "train": "zbiór uczący",
    "validation": "zbiór walidacyjny",
    "test": "zbiór testowy",
}


def split_name(split: str) -> str:
    """Name of a split for a legend, capitalized: `Zbiór uczący`."""
    return SPLIT_NAMES[split].capitalize()


def class_name(label: int) -> str:
    """Legend entry of a class with its original label."""
    return f"Klasa {label}"


def variable(name: str) -> str:
    """Axis label of a data column: a short name in italics (`$x$`), longer ones as they are."""
    if name.isidentifier() and len(name) <= 2:
        return f"${name}$"
    return name
