# Compute Layer

Rust crate that trains the multilayer perceptron and writes training artifacts. It is split into a library (`src/lib.rs`) and a thin `compute` binary (`src/main.rs`) that provides the CLI.

The Rust version is pinned in `rust-toolchain.toml`.

## Usage

Every command takes an experiment directory that contains `config.toml`:

```bash
compute validate <experiment-dir>   # check the config and dataset files, print a summary
compute train <experiment-dir>      # prepare the data and train the network (training not implemented yet)
```

Relative paths, both of the experiment directory and of the datasets in the config, are resolved against the current working directory. Run the binary from the repository root:

```bash
cargo run --manifest-path src/compute/Cargo.toml -- validate data/experiments/classification/example-three-gauss
```

Invalid configs are rejected with diagnostics that point at the offending line of `config.toml`. All problems are reported at once. Suspicious but valid settings, such as Xavier initialization with ReLU, produce warnings.

## Experiment config

`config.toml` is a TOML file with the sections below. Every field is required, except `data.train_subset` and the Adam parameters `beta1`, `beta2` and `epsilon` (defaults 0.9, 0.999 and 1e-8). Unknown fields are rejected. See the commented examples in `data/experiments/classification/example-three-gauss/` and `data/experiments/regression/example-cube/`.

| Key | Values |
|---|---|
| `task` | `classification`, `regression` (MNIST is classification) |
| `data.train`, `data.test` | CSV paths; both files must have the same columns |
| `data.target` | target column name; all other columns are features |
| `data.input_normalization`, `data.target_normalization` | `none`, `standard`, `min_max`; the target must be `none` for classification |
| `data.train_subset` | optional number of training samples drawn at random |
| `data.validation_fraction` | in `[0, 1)`; `0.0` disables validation |
| `model.hidden_layers` | 0 to 4 positive layer sizes, e.g. `[16, 16]` |
| `model.bias` | `true`, `false` |
| `model.hidden_activation` | `{ kind = "sigmoid" }`, `tanh`, `relu`, `{ kind = "leaky_relu", alpha = 0.01 }` |
| `model.output_activation` | `identity` (regression), `softmax`, `sigmoid` |
| `model.init` | `{ kind = "xavier" }`, `he`, `{ kind = "uniform", limit = 0.5 }` |
| `optimizer.kind` | `sgd`, `momentum` (+ `momentum`), `adam` (+ optional `beta1`, `beta2`, `epsilon`); all need `learning_rate` |
| `training.loss` | `cross_entropy` (needs `softmax`), `mse`, `mae` (regression only) |
| `training.epochs`, `training.batch_size`, `training.seed` | positive integers; the seed determines every random choice |
| `logging.loss_every`, `logging.weights_every` | intervals in iterations (batches) |
| `logging.max_weights_per_layer` | upper bound on recorded weights per layer |
| `output.grid_resolution`, `output.grid_margin` | prediction grid for 1D and 2D inputs |

## Data preparation

`compute train` loads the datasets before training:

1. Both CSV files are read whole. The header names the columns; every value must be a finite number.
2. Classification: labels must be integers. The classes are the distinct labels of the whole training file, sorted, so labels `1, 2, 3` map to network outputs `0, 1, 2` and MNIST labels `0..=9` map to themselves. A test label missing from the training file is an error. Targets become one-hot vectors.
3. `train_subset` samples are drawn from the training file, then `validation_fraction` of them is held out for validation (rounded to the nearest sample count). For classification both draws are stratified: every class keeps its share of samples. Both use a random stream derived from `training.seed`, separate from other random choices, so the split does not change when e.g. the network changes.
4. Normalization statistics are computed per column on the remaining training samples only, then applied to the training, validation and test sets. `standard` subtracts the mean and divides by the standard deviation, `min_max` maps the training range to `[0, 1]`. A constant column is only shifted to 0.

A `train_subset` larger than the training set, or a `validation_fraction` that leaves the validation or training set empty, is reported at its line in `config.toml`.

## Commands

```bash
cargo build --release                        # build
cargo run --release -- --help                # show the CLI help
cargo test                                   # run all tests
cargo test <test_name>                       # run a single test
cargo clippy --all-targets -- -D warnings    # lint (as in CI)
cargo fmt                                    # format
```
