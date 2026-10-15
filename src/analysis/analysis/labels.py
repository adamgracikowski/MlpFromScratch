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


ITERATION = "Iteracja"
EPOCH = "Epoka"
ACCURACY = "Dokładność [%]"
MSE = "MSE (skala oryginalna)"
MOVING_AVERAGE = "Średnia krocząca (1 epoka)"
WEIGHT_VALUE = "Wartość parametru"
WEIGHTS = "Wagi $w_{ij}$"
BIASES = "Biasy $b_j$"
SAMPLED_WEIGHTS = "Parametry"
NORM = "Norma"
WEIGHT_NORM = "$\\|W\\|_F$"
BIAS_NORM = "$\\|b\\|_2$"

MISCLASSIFIED = "Błędnie sklasyfikowane"
TARGET_FUNCTION = "Funkcja docelowa (zbiór testowy)"
NETWORK_OUTPUT = "Predykcja sieci"
RESIDUAL = r"Reszta $\hat{y} - y$"
OUTSIDE_TRAINING_RANGE = "Poza zakresem zbioru uczącego"
TRUE_CLASS = "Prawdziwa klasa"
PREDICTED_CLASS = "Przewidziana klasa"
ROW_SHARE = "Udział w klasie prawdziwej [%]"

LOSS_NAMES = {
    "cross_entropy": "entropia krzyżowa",
    "mse": "MSE",
    "mae": "MAE",
}


def loss_label(loss: str) -> str:
    """Axis label of the training loss, e.g. `Strata (entropia krzyżowa)`."""
    return f"Strata ({LOSS_NAMES[loss]})"


def loss_window(loss_every: int) -> str:
    """Legend entry of the recorded loss: one batch or the mean of a window of batches."""
    if loss_every == 1:
        return "Strata minibatcha"
    return f"Średnia z {loss_every} minibatchy"


def best_epoch(epoch: int) -> str:
    """Legend entry of the epoch whose weights were kept by early stopping."""
    return f"Najlepsza epoka ({epoch})"


def layer_name(layer: int, layer_count: int) -> str:
    """Name of a layer numbered from 0: hidden layers from 1, then the output layer."""
    if layer == layer_count - 1:
        return "Warstwa wyjściowa"
    return f"Warstwa ukryta {layer + 1}"


def variable(name: str) -> str:
    """Axis label of a data column: a short name in italics (`$x$`), longer ones as they are."""
    if name.isidentifier() and len(name) <= 2:
        return f"${name}$"
    return name
