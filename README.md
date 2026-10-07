# MlpFromScratch

High-performance Multilayer Perceptron built from scratch without ML frameworks for the Neural Networks course. Features a custom compute engine in Rust and data visualization in Python.

The network (forward pass, backpropagation, SGD and its variants) is implemented by hand. It is used for regression (f: R → R), classification of points in R², and MNIST digit recognition.

## Architecture

The project has two layers that communicate only through files on disk:

- **Compute layer** (`src/compute/`, Rust): trains the network and writes raw artifacts. It does no plotting.
- **Presentation layer** (`src/analysis/`, Python): reads artifacts and produces plots and statistics. It does no training.

Each experiment lives in `data/experiments/<task>/<experiment-name>/`, where `<task>` is `classification`, `regression` or `mnist`:

1. A config file in the experiment folder holds all parameters, including the RNG seed.
2. The compute binary reads the config, trains the network and writes results into `artifacts/`.
3. The analysis scripts read `artifacts/` and write plots and statistics into `plots/`.

Configs and plots are committed. Artifacts are git-ignored.

## Repository layout

```txt
data/
├── datasets/        # classification, regression (CSV) and mnist (downloaded)
└── experiments/     # one folder per experiment
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

Compute layer:

```bash
cd src/compute
cargo build --release
cargo run --release -- --help
cargo test
```

Presentation layer:

```bash
cd src/analysis
uv sync
uv run main.py --help
uv run pytest
```

See [`src/compute/README.md`](src/compute/README.md) and [`src/analysis/README.md`](src/analysis/README.md) for details.
