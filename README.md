# MlpFromScratch

High-performance Multilayer Perceptron built from scratch without ML frameworks for the Neural Networks course. Features a custom compute engine in Rust and data visualization in Python.

The network (forward pass, backpropagation, SGD and its variants) is implemented by hand. It is used for regression (f: R → R), classification of points in R², and MNIST digit recognition.

## Architecture

The project has two layers that communicate only through files on disk:

- **Compute layer** (`src/compute/`, Rust): trains the network and writes raw artifacts. It does no plotting.
- **Presentation layer** (`src/analysis/`, Python): reads artifacts and produces plots and statistics. It does no training.

Each experiment lives in `data/experiments/<task>/<experiment-name>/`, where `<task>` is `classification`, `regression` or `mnist`:

1. `config.toml` in the experiment folder holds all parameters, including the RNG seed.
2. The compute binary reads the config, trains the network and writes results into `artifacts/`.
3. The analysis scripts read `artifacts/` and write plots and statistics into `plots/`.

Configs and plots are committed. Artifacts are git-ignored.

A study compares many experiments: `study.toml` in `data/experiments/<task>/<study-name>/` holds a base config, axes of parameters to vary and a list of seeds. `compute generate` writes one experiment folder per combination of levels and seed into `runs/` (generated, git-ignored: `study.toml` and the code version reproduce it), `compute train` trains them in parallel, and `aggregate` writes tables over seeds (mean, standard deviation), comparison plots and `summary.md` into `results/`. Examples: `data/experiments/classification/example-study/` and `data/experiments/regression/example-study/`.

## Repository layout

```txt
data/
├── datasets/        # classification, regression (CSV) and mnist (downloaded)
└── experiments/     # one folder per experiment
scripts/             # run-studies.ps1: runs all studies, resumable
src/
├── compute/         # Rust crate (library + `compute` binary)
└── analysis/        # Python package managed with uv
.github/workflows/   # CI: fmt, clippy, tests, ruff, pytest
```

## Prerequisites

- [rustup](https://rustup.rs/). The toolchain version is pinned in `src/compute/rust-toolchain.toml` and is installed automatically.
- [uv](https://docs.astral.sh/uv/). It installs Python 3.13 and the dependencies.
- MNIST is not in the repository. See [`data/datasets/mnist/README.md`](data/datasets/mnist/README.md).

## Usage

Compute layer, run from the repository root (dataset paths in configs are relative to the current working directory):

```bash
cargo build --release --manifest-path src/compute/Cargo.toml
cargo run --release --manifest-path src/compute/Cargo.toml -- validate data/experiments/classification/example-three-gauss
cargo run --release --manifest-path src/compute/Cargo.toml -- train data/experiments/classification/example-three-gauss
```

A study, from the repository root:

```bash
cargo run --release --manifest-path src/compute/Cargo.toml -- generate data/experiments/classification/example-study
cargo run --release --manifest-path src/compute/Cargo.toml -- train data/experiments/classification/example-study --jobs 8
uv run --project src/analysis src/analysis/main.py aggregate data/experiments/classification/example-study --plot-runs
```

The studies of part one are in `data/experiments/<task>/<dataset>-<factor>/`, one per dataset and studied factor, each with the factor and the training set size (100, 500, 1000, 10000; fewer epochs for larger sets) as axes and 10 seeds:

| Factor | Levels | Studies |
|---|---|---|
| `activations` | hidden activation: sigmoid, tanh (Xavier initialization), ReLU, leaky ReLU (He) | `classification/{simple,three-gauss}-activations`, `regression/{activation,cube}-activations` |
| `architectures` | hidden layers: none, 4, 16, 64, 16-16, 64-64, 16-16-16, 16-16-16-16, 64-64-64-64 | `classification/{simple,three-gauss}-architectures`, `regression/{activation,cube}-architectures` |
| `losses` | cross-entropy and MSE with softmax (classification), MSE and MAE (regression) | `classification/{simple,three-gauss}-losses`, `regression/{activation,cube}-losses` |

The other settings are shared: tanh, 16-16 (classification) or 32-32 (regression) hidden neurons, Adam with learning rate 0.003, batch size 32.

All studies of part one, one after another (PowerShell 7, from any directory):

```powershell
pwsh -File scripts/run-studies.ps1                  # every study, the cheapest first
pwsh -File scripts/run-studies.ps1 -Jobs 6 data/experiments/regression/cube-losses
pwsh -File scripts/run-studies.ps1 -ForceAggregate  # aggregate again, e.g. after a plotting change
```

For every study the script runs `generate`, `train --skip-trained` and `aggregate --plot-runs`, printing their output and appending it to `logs/run-studies-<date>-<time>.log`; at the end it prints a table of the studies (`done`, `incomplete` when some runs failed, `failed`) and exits with 1 unless all are done. It can be stopped at any time (Ctrl+C) and started again: trained runs are skipped (a run counts as trained once `artifacts/metadata.json`, written last, holds its current config), so only the runs in progress are lost, and aggregation is skipped while `results/summary.md` is newer than `study.toml`, `runs/index.csv` and every `metadata.json` of the study. While it runs, Windows does not go to sleep; closing the lid may still suspend the laptop, depending on the power settings.

Tests, from `src/compute/`:

```bash
cargo test
```

Presentation layer, also from the repository root (plots and `summary.md` go to the `plots/` folder of the experiment):

```bash
uv run --project src/analysis src/analysis/main.py plot data/experiments/classification/example-three-gauss
```

Tests, from `src/analysis/`:

```bash
uv sync
uv run pytest
```

See [`src/compute/README.md`](src/compute/README.md) and [`src/analysis/README.md`](src/analysis/README.md) for details.
