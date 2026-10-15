"""Polish texts shown on the plots (the report is written in Polish).

This is the only module with non-ASCII text: Polish words need their diacritics.
It holds the texts of the plots and of `summary.md`.
"""

SPLIT_ADJECTIVES = {"train": "uczący", "validation": "walidacyjny", "test": "testowy"}
SPLIT_NAMES = {split: f"zbiór {adjective}" for split, adjective in SPLIT_ADJECTIVES.items()}


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


# summary.md

SUMMARY_TITLE = "Eksperyment"
SECTION_CONFIG = "Konfiguracja"
SECTION_TRAINING = "Przebieg uczenia"
SECTION_METRICS = "Metryki końcowe"
SECTION_VERSION = "Wersja kodu"
PARAMETER = "Parametr"
VALUE = "Wartość"
SPLIT = "Zbiór"
LOSS = "Strata"
ACCURACY_COLUMN = "Dokładność"
MSE_COLUMN = "MSE"
MAE_COLUMN = "MAE"

TASK = "Zadanie"
TRAIN_FILE = "Plik uczący"
TEST_FILE = "Plik testowy"
TRAIN_SUBSET = "Podzbiór uczący"
VALIDATION_FRACTION = "Część walidacyjna"
SAMPLES = "Liczności zbiorów"
NORMALIZATION = "Normalizacja wejść / celu"
ARCHITECTURE = "Architektura"
ACTIVATIONS = "Aktywacje"
INIT = "Inicjalizacja wag"
LOSS_FUNCTION = "Funkcja straty"
OPTIMIZER = "Optymalizator"
EPOCHS_AND_BATCH = "Epoki, rozmiar batcha"
EARLY_STOPPING = "Early stopping"
SEED = "Ziarno"
STOP = "Zakończenie"
EPOCHS_AND_ITERATIONS = "Epoki, iteracje"
BEST_EPOCH = "Najlepsza epoka"
TRAINING_TIME = "Czas uczenia"
NO = "nie"

LAYER_SEPARATOR = "-"  # architecture as layer sizes: 2-16-16-3
TASK_NAMES = {"classification": "klasyfikacja", "regression": "regresja"}
NORMALIZATION_NAMES = {"none": "brak", "standard": "standaryzacja", "min_max": "min-max"}
OPTIMIZER_NAMES = {"sgd": "SGD", "momentum": "SGD z momentum", "adam": "Adam"}
METRICS_NOTE = "Metryki sieci zwróconej przez uczenie (przy early stoppingu: z najlepszej epoki)."
REGRESSION_METRICS_NOTE = "MSE i MAE w oryginalnej skali celu."


def plural(count: int, one: str, few: str, many: str) -> str:
    """Polish noun form after a number: 1 parametr, 2-4 parametry (not 12-14), 5+ parametrów."""
    if count == 1:
        return one
    if count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
        return few
    return many


def parameter_count(count: int, bias: bool) -> str:
    """Parameter count of the network, e.g. `371 parametrów, z biasem`."""
    noun = plural(count, "parametr", "parametry", "parametrów")
    return f"{count} {noun}, {'z biasem' if bias else 'bez biasu'}"


def train_subset(size: int) -> str:
    return f"{size} {plural(size, 'próbka', 'próbki', 'próbek')}"


def validation_fraction(percent: str) -> str:
    return f"{percent} zbioru uczącego"


def stop_reason(stop: dict) -> str:
    """How training ended, from `result.stop` of the metadata."""
    reason = stop["reason"]
    if reason == "completed":
        return "wszystkie epoki"
    if reason == "early_stopped":
        return "early stopping"
    return f"dywergencja (epoka {stop['epoch']}, iteracja {stop['iteration']})"


def seconds(text: str) -> str:
    return f"{text} s"


def code_version(package: str, commit: str, dirty: bool) -> str:
    """Version of the compute layer that trained the network."""
    changes = " (z niezatwierdzonymi zmianami)" if dirty else ""
    return f"`compute` {package}, commit `{commit}`{changes}"
