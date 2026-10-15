"""Shared fixtures: synthetic experiments laid out like the output of `compute train`."""

import json
import math
from collections.abc import Callable
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
import pytest

matplotlib.use("Agg")

TRAIN_ROWS = 40
TEST_ROWS = 30
EPOCHS = 4
BATCH_SIZE = 16
HIDDEN = 4
MAX_WEIGHTS = 8
GRID_RESOLUTION = 5


def write_experiment(
    root: Path,
    *,
    task: str = "classification",
    features: int = 2,
    validation: bool = False,
    early_stopped: bool = False,
    diverged: bool = False,
    bias: bool = True,
) -> Path:
    """Write datasets and `artifacts/` of a small synthetic run; return the experiment dir."""
    rng = np.random.default_rng(0)
    classification = task == "classification"
    feature_names = ["x", "y"][:features] if features <= 2 else [f"f{i}" for i in range(features)]
    target = "cls" if classification else "y"
    classes = [1, 2, 3] if classification else None
    outputs = len(classes) if classification else 1

    datasets = root / "datasets"
    datasets.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, rows in (("train", TRAIN_ROWS), ("test", TEST_ROWS)):
        columns = {name: rng.normal(size=rows) for name in feature_names}
        if classification:
            columns[target] = rng.integers(1, 4, size=rows)
        else:
            columns[target] = columns["x"] ** 3 * 100 if "x" in columns else rng.normal(size=rows)
        paths[split] = datasets / f"{task}.{features}.{split}.csv"
        pl.DataFrame(columns).write_csv(paths[split])

    experiment = root / "experiment"
    artifacts = experiment / "artifacts"
    artifacts.mkdir(parents=True)

    # Splits: validation is held out from the training file.
    permutation = rng.permutation(TRAIN_ROWS)
    validation_rows = permutation[: TRAIN_ROWS // 5] if validation else permutation[:0]
    split_rows = {
        "train": np.sort(permutation[len(validation_rows) :]),
        "validation": np.sort(validation_rows),
        "test": np.arange(TEST_ROWS),
    }
    splits = ["train", "validation", "test"] if validation else ["train", "test"]
    train_csv = pl.read_csv(paths["train"])
    test_csv = pl.read_csv(paths["test"])
    for split in splits:
        rows = split_rows[split]
        source = test_csv if split == "test" else train_csv
        directory = artifacts / "predictions" / split
        directory.mkdir(parents=True)
        _save(directory / "rows.npy", rows.astype(np.uint64))
        truth = source[target].to_numpy()[rows]
        if classification:
            probabilities = rng.dirichlet(np.ones(outputs), size=len(rows))
            _save(directory / "outputs.npy", probabilities)
            _save(directory / "labels.npy", truth.astype(np.int64))
            predicted = np.asarray(classes)[probabilities.argmax(axis=1)]
            _save(directory / "predicted.npy", predicted.astype(np.int64))
        else:
            _save(directory / "outputs.npy", (truth + rng.normal(size=len(rows)))[:, None])
            _save(directory / "targets.npy", truth.astype(np.float64)[:, None])

    # Epoch metrics and loss history.
    train_samples = len(split_rows["train"])
    per_epoch = math.ceil(train_samples / BATCH_SIZE)
    epochs = EPOCHS
    iterations = epochs * per_epoch
    columns = {"epoch": np.arange(epochs + 1), "iteration": np.arange(epochs + 1) * per_epoch}
    metric_names = ["accuracy"] if classification else ["mse", "mae"]
    for split in splits:
        columns[f"{split}_loss"] = np.linspace(1.0, 0.1, epochs + 1)
        for metric in metric_names:
            columns[f"{split}_{metric}"] = np.linspace(0.3, 0.9, epochs + 1)
    if diverged:
        for name in columns:
            if name not in ("epoch", "iteration"):
                columns[name][-1] = math.inf if name.endswith("loss") else math.nan
    pl.DataFrame(columns).write_csv(artifacts / "epochs.csv")

    history = artifacts / "history"
    history.mkdir()
    loss = np.exp(-np.arange(iterations) / iterations) + rng.uniform(0, 0.1, size=iterations)
    if diverged:
        loss[-2:] = [math.inf, math.nan]
    _save(history / "iteration.npy", np.arange(1, iterations + 1, dtype=np.uint64))
    _save(history / "epoch.npy", (np.arange(iterations) // per_epoch + 1).astype(np.uint64))
    _save(history / "loss.npy", loss)

    # Weight snapshots: every 2 iterations plus the last one.
    layer_sizes = [features, HIDDEN, outputs]
    snapshots = np.unique(np.append(np.arange(0, iterations + 1, 2), iterations))
    weights = artifacts / "weights"
    weights.mkdir()
    _save(weights / "iteration.npy", snapshots.astype(np.uint64))
    sampled = []
    for layer, (fan_in, fan_out) in enumerate(zip(layer_sizes, layer_sizes[1:], strict=False)):
        pool = [(row, column) for row in range(fan_in + bias) for column in range(fan_out)]
        if len(pool) > MAX_WEIGHTS:
            chosen = np.sort(rng.choice(len(pool), size=MAX_WEIGHTS, replace=False))
            pool = [pool[i] for i in chosen]
        sampled.append(len(pool))
        _save(weights / f"layer_{layer}_indices.npy", np.asarray(pool, dtype=np.uint64))
        drift = np.cumsum(rng.normal(scale=0.1, size=(len(snapshots), len(pool))), axis=0)
        _save(weights / f"layer_{layer}_values.npy", drift)
    layer_count = len(layer_sizes) - 1
    norms = 1 + np.cumsum(rng.uniform(size=(len(snapshots), layer_count)), axis=0) / 10
    _save(weights / "weight_norms.npy", norms)
    _save(weights / "bias_norms.npy", norms / 2 if bias else np.full_like(norms, np.nan))

    grid = None
    if features <= 2:
        grid_dir = artifacts / "grid"
        grid_dir.mkdir()
        axes = [np.linspace(-3, 3, GRID_RESOLUTION) for _ in range(features)]
        for name, axis in zip(("x.npy", "y.npy"), axes, strict=False):
            _save(grid_dir / name, axis)
        shape = (GRID_RESOLUTION,) * features
        if classification:
            values = rng.dirichlet(np.ones(outputs), size=shape)
        else:
            values = (axes[0] ** 3 * 100)[:, None]
        _save(grid_dir / "outputs.npy", values)
        grid = {"resolution": GRID_RESOLUTION, "ranges": [[-3.0, 3.0]] * features}

    stop = {"reason": "completed"}
    if early_stopped:
        stop = {"reason": "early_stopped"}
    if diverged:
        stop = {"reason": "diverged", "epoch": epochs, "iteration": iterations - 1}
    final = {"loss": 0.1, "accuracy": 0.9, "mse": None, "mae": None}
    if not classification:
        final = {"loss": 0.1, "accuracy": None, "mse": 1234.5, "mae": 12.5}
    metadata = {
        "version": {"package": "0.1.0", "commit": "0123456789abcdef", "dirty": False},
        "started_at": 1792076037,
        "training_seconds": 0.25,
        "config": {
            "task": task,
            "data": {
                "train": paths["train"].as_posix(),
                "test": paths["test"].as_posix(),
                "target": target,
                "input_normalization": "standard",
                "target_normalization": "none" if classification else "standard",
                "validation_fraction": 0.2 if validation else 0.0,
            },
            "model": {
                "hidden_layers": [HIDDEN],
                "bias": bias,
                "hidden_activation": {"kind": "leaky_relu", "alpha": 0.01},
                "output_activation": "softmax" if classification else "identity",
                "init": {"kind": "he"},
            },
            "optimizer": {"kind": "momentum", "learning_rate": 0.01, "momentum": 0.9},
            "training": {
                "loss": "cross_entropy" if classification else "mse",
                "epochs": epochs,
                "batch_size": BATCH_SIZE,
                "seed": 42,
                **(
                    {"early_stopping": {"patience": 2, "min_delta": 0.0001}}
                    if early_stopped
                    else {}
                ),
            },
            "logging": {"loss_every": 1, "weights_every": 2, "max_weights_per_layer": MAX_WEIGHTS},
            "output": {"grid_resolution": GRID_RESOLUTION, "grid_margin": 0.1},
        },
        "data": {
            "features": feature_names,
            "target": target,
            "samples": {
                "train": train_samples,
                "validation": len(split_rows["validation"]) if validation else None,
                "test": TEST_ROWS,
            },
            "classes": classes,
            "input_scaler": {
                "normalization": "standard",
                "offset": [0.0] * features,
                "scale": [1.0] * features,
            },
            "target_scaler": None,
        },
        "network": {
            "layer_sizes": layer_sizes,
            "activations": ["leaky_relu(alpha=0.01)", "softmax" if classification else "identity"],
            "bias": bias,
            "parameter_count": sum(
                (fan_in + bias) * fan_out
                for fan_in, fan_out in zip(layer_sizes, layer_sizes[1:], strict=False)
            ),
        },
        "result": {
            "stop": stop,
            "epochs": epochs,
            "iterations": iterations,
            "best_epoch": 2 if early_stopped else None,
            "metrics": {
                "train": final,
                "validation": final if validation else None,
                "test": final,
            },
        },
        "history": {
            "loss_every": 1,
            "loss_points": iterations,
            "weights_every": 2,
            "snapshots": len(snapshots),
            "sampled_parameters": sampled,
        },
        "grid": grid,
    }
    (artifacts / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return experiment


def _save(path: Path, array: np.ndarray) -> None:
    np.save(path, np.ascontiguousarray(array))


@pytest.fixture
def make_experiment(tmp_path: Path) -> Callable[..., Path]:
    """Factory writing a synthetic experiment into a fresh temporary directory."""
    counter = iter(range(1000))

    def make(**options: object) -> Path:
        return write_experiment(tmp_path / f"run{next(counter)}", **options)

    return make
