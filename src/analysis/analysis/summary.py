"""`summary.md`: configuration, course of training and final metrics in Markdown tables."""

from pathlib import Path
from typing import Any

from analysis import labels
from analysis.experiment import SPLITS, Experiment
from analysis.formatting import format_percent, format_value

SUMMARY_FILE = "summary.md"


def write_summary(experiment: Experiment, directory: Path) -> Path:
    """Write `<directory>/summary.md` and return its path."""
    path = directory / SUMMARY_FILE
    path.write_text(summary_markdown(experiment), encoding="utf-8", newline="\n")
    return path


def summary_markdown(experiment: Experiment) -> str:
    """The summary of one experiment as Markdown (Polish, like the report)."""
    metadata = experiment.metadata
    sections = [
        f"# {labels.SUMMARY_TITLE} `{experiment.directory.name}`",
        f"## {labels.SECTION_CONFIG}",
        markdown_table([labels.PARAMETER, labels.VALUE], _config_rows(metadata)),
        f"## {labels.SECTION_TRAINING}",
        markdown_table([labels.PARAMETER, labels.VALUE], _training_rows(metadata)),
        f"## {labels.SECTION_METRICS}",
        markdown_table(*_metric_rows(experiment)),
        labels.METRICS_NOTE
        + ("" if experiment.is_classification else " " + labels.REGRESSION_METRICS_NOTE),
        f"## {labels.SECTION_VERSION}",
        labels.code_version(
            metadata["version"]["package"],
            metadata["version"]["commit"][:7],
            metadata["version"]["dirty"],
        ),
    ]
    return "\n\n".join(sections) + "\n"


def _config_rows(metadata: dict[str, Any]) -> list[list[str]]:
    config = metadata["config"]
    data_config = config["data"]
    model = config["model"]
    training = config["training"]
    network = metadata["network"]

    rows = [
        [labels.TASK, labels.TASK_NAMES[config["task"]]],
        [labels.TRAIN_FILE, f"`{data_config['train']}`"],
        [labels.TEST_FILE, f"`{data_config['test']}`"],
    ]
    if data_config.get("train_subset") is not None:
        rows.append([labels.TRAIN_SUBSET, labels.train_subset(data_config["train_subset"])])
    if data_config["validation_fraction"] > 0:
        percent = format_percent(data_config["validation_fraction"], 0)
        rows.append([labels.VALIDATION_FRACTION, labels.validation_fraction(percent)])
    samples = metadata["data"]["samples"]
    rows += [
        [
            labels.SAMPLES,
            ", ".join(
                f"{labels.SPLIT_ADJECTIVES[split]}: {samples[split]}"
                for split in SPLITS
                if samples[split] is not None
            ),
        ],
        [
            labels.NORMALIZATION,
            f"{labels.NORMALIZATION_NAMES[data_config['input_normalization']]} / "
            f"{labels.NORMALIZATION_NAMES[data_config['target_normalization']]}",
        ],
        [
            labels.ARCHITECTURE,
            labels.LAYER_SEPARATOR.join(str(size) for size in network["layer_sizes"])
            + f" ({labels.parameter_count(network['parameter_count'], network['bias'])})",
        ],
        [labels.ACTIVATIONS, ", ".join(f"`{name}`" for name in network["activations"])],
        [labels.INIT, tagged(model["init"])],
        [labels.LOSS_FUNCTION, labels.LOSS_NAMES[training["loss"]]],
        [labels.OPTIMIZER, _optimizer(config["optimizer"])],
        [labels.EPOCHS_AND_BATCH, f"{training['epochs']}, {training['batch_size']}"],
        [
            labels.EARLY_STOPPING,
            _parameters(training["early_stopping"])
            if training.get("early_stopping")
            else labels.NO,
        ],
        [labels.SEED, str(training["seed"])],
    ]
    return rows


def _training_rows(metadata: dict[str, Any]) -> list[list[str]]:
    result = metadata["result"]
    rows = [
        [labels.STOP, labels.stop_reason(result["stop"])],
        [labels.EPOCHS_AND_ITERATIONS, f"{result['epochs']}, {result['iterations']}"],
    ]
    if result["best_epoch"] is not None:
        rows.append([labels.BEST_EPOCH, str(result["best_epoch"])])
    rows.append(
        [labels.TRAINING_TIME, labels.seconds(format_value(metadata["training_seconds"], 3))]
    )
    return rows


def _metric_rows(experiment: Experiment) -> tuple[list[str], list[list[str]]]:
    metrics = experiment.metadata["result"]["metrics"]
    if experiment.is_classification:
        header = [labels.SPLIT, labels.LOSS, labels.ACCURACY_COLUMN]
    else:
        header = [labels.SPLIT, labels.LOSS, labels.MSE_COLUMN, labels.MAE_COLUMN]
    rows = []
    for split in SPLITS:
        split_metrics = metrics.get(split)
        if split_metrics is None:
            continue
        row = [labels.SPLIT_ADJECTIVES[split], format_value(split_metrics["loss"])]
        if experiment.is_classification:
            accuracy = split_metrics["accuracy"]
            row.append(format_value(None) if accuracy is None else format_percent(accuracy, 2))
        else:
            row += [format_value(split_metrics["mse"]), format_value(split_metrics["mae"])]
        rows.append(row)
    return header, rows


def _optimizer(optimizer: dict[str, Any]) -> str:
    return f"{labels.OPTIMIZER_NAMES[optimizer['kind']]}: {_parameters(optimizer)}"


def tagged(value: dict[str, Any]) -> str:
    """A config enum with a `kind` tag: `` `he` `` or `` `uniform` (`limit` = 0,5) ``."""
    parameters = _parameters(value)
    return f"`{value['kind']}`" + (f" ({parameters})" if parameters else "")


def _parameters(values: dict[str, Any]) -> str:
    """Config parameters except the `kind` tag: `` `learning_rate` = 0,01, ... ``."""
    return ", ".join(
        f"`{name}` = {config_value(value)}" for name, value in values.items() if name != "kind"
    )


def config_value(value: Any) -> str:
    if isinstance(value, int):  # also bool
        return str(value)
    if isinstance(value, float):
        return format_value(value)
    return f"`{value}`"


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)
