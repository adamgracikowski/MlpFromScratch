# Compute Layer

Rust crate that trains the multilayer perceptron and writes training artifacts for the presentation layer (`src/analysis/`). It does no plotting. It is split into a library (`src/lib.rs`) and a thin `compute` binary (`src/main.rs`) that provides the CLI. See the [main README](../../README.md) for an overview of the project.

The Rust version (1.88.0, edition 2024) is pinned in `rust-toolchain.toml`; dependency versions stay compatible with it.

## Usage

`validate` takes an experiment directory that contains `config.toml`; `train` takes one or more of them, or directories to search for them; `generate` takes a study directory that contains `study.toml` (see [Studies](#studies)):

```bash
compute generate <study-dir>        # write one experiment directory per variant and seed of a study
compute validate <experiment-dir>   # check the config and dataset files, print a summary
compute train <experiment-dir>      # prepare the data, train the network, print progress and results
compute train <experiment-dir> --progress-every 10   # print metrics every 10 epochs (0: only the summary)
compute train <experiment-dir> --no-artifacts        # train without writing artifacts
compute train <study-dir> --jobs 8                   # train every run of a study, 8 at a time
compute train <study-dir> --skip-trained             # train only runs without artifacts of their current config
```

`train` treats a directory with `config.toml` as one experiment and any other directory as every experiment below it (sorted; `artifacts/`, `plots/` and `results/` are not searched). A single experiment prints its config, progress and results. Several are trained in parallel (`--jobs`, default: the number of CPUs) with one line per finished run, e.g. `[3/18] runs/tanh__simple-100/seed-1: completed after 200 epoch(s), test loss 0.0166 acc 99.00% (0.08 s)`, and a summary. A failed run does not stop the others; its error is printed at the end and the command exits with code 1. Every run has its own seeded random streams, so the results do not depend on `--jobs`.

`--skip-trained` skips a run when `artifacts/metadata.json` exists and holds the same config; a diverged run counts as trained. Without it, every run is trained again.

Relative paths, both of the experiment directory and of the datasets in the config, are resolved against the current working directory. Run the binary from the repository root:

```bash
cargo run --manifest-path src/compute/Cargo.toml -- validate data/experiments/classification/example-three-gauss
```

Invalid configs are rejected with diagnostics that point at the offending line of `config.toml`. All problems are reported at once. Suspicious but valid settings, such as Xavier initialization with ReLU, produce warnings.

## Experiment config

`config.toml` is a TOML file with the sections below. Every field is required, except `data.train_subset`, `training.early_stopping` and the Adam parameters `beta1`, `beta2` and `epsilon` (defaults 0.9, 0.999 and 1e-8). Unknown fields are rejected. See the commented examples in `data/experiments/classification/example-three-gauss/` and `data/experiments/regression/example-cube/`.

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
| `training.early_stopping` | optional `{ patience = 20, min_delta = 1e-4 }` (`min_delta` defaults to 0); needs `data.validation_fraction > 0` |
| `logging.loss_every`, `logging.weights_every` | intervals in iterations (batches), see [Artifacts](#artifacts) |
| `logging.max_weights_per_layer` | upper bound on recorded parameters per layer |
| `output.grid_resolution`, `output.grid_margin` | prediction grid for 1D and 2D inputs |

## Studies

A study is a grid of experiments described by `<study-dir>/study.toml`, e.g. `data/experiments/classification/<study>/study.toml`. It holds a complete base config, the seeds and the axes to vary. See the commented examples in `data/experiments/classification/example-study/` and `data/experiments/regression/example-study/`; a test checks that every config they generate is valid. Only `study.toml` is committed: `runs/` is generated and git-ignored.

```toml
description = "Hidden activation on two datasets"   # optional, copied into the study summary
seeds = [1, 2, 3]                                    # every variant runs once per seed

[[axes]]                                             # shorthand: one key, one level per value
name = "activation"
key = "model.hidden_activation"
values = [{ kind = "sigmoid" }, { kind = "tanh" }, { kind = "relu" }]

[[axes]]                                             # full form: named levels setting several keys
name = "dataset"
levels = [
  { name = "simple-100", set = { "data.train" = "data/datasets/classification/data.simple.train.100.csv", "data.test" = "data/datasets/classification/data.simple.test.100.csv" } },
  { name = "three-gauss-500", set = { "data.train" = "data/datasets/classification/data.three_gauss.train.500.csv", "data.test" = "data/datasets/classification/data.three_gauss.test.500.csv" } },
]

[report]                                             # optional: placement of the axes on the comparison plots
x = "activation"                                     # categories on the x axis
panels = ["dataset"]                                 # one panel per level; `color = "..."` colors an axis within a panel
labels = { dataset = "Zbiór danych", "three-gauss-500" = "three_gauss (500)" }  # optional: plot texts for axis or level names
split = "test"                                       # optional: split whose metric compares the variants: train | validation | test (default)

[base]                                               # a complete config.toml under `base.`, without training.seed
task = "classification"

[base.data]
# ...
```

`compute generate <study-dir>` combines the levels of all axes (cartesian product; the first axis varies slowest) and writes one experiment directory per variant and seed:

```txt
<study-dir>/
├── study.toml
└── runs/
    ├── index.csv                       # dir,variant,seed,<axis>,...: one row per run
    ├── sigmoid__simple-100/            # level names joined by `__` (`base` without axes)
    │   ├── seed-1/config.toml          # an ordinary experiment directory
    │   └── ...
    └── ...
```

- A level replaces the whole value under its dotted key, e.g. `{ kind = "leaky_relu", alpha = 0.01 }` for `model.hidden_activation`. In the full form, quote the dotted keys: `"data.train" = ...`. The parent table must exist in `[base]`.
- Shorthand level names come from the values: strings as they are, numbers and booleans as written, arrays joined by `-` (`[]` is `0`, `[64, 16]` is `64-16`), tables as their `kind` followed by the other values (`leaky_relu-0.01`). Names may use letters, digits, `.`, `_` and `-`; values such as paths need the full form with explicit names.
- Every problem of `study.toml` is reported at its line: empty or duplicate seeds, duplicate axis or level names, an axis with both or neither form, a key without a parent table, `training.seed` in `[base]` or in an axis, two axes setting the same value, report roles that name no axis or one axis twice, `labels` keys that are neither an axis nor a level name, and `split = "validation"` with a variant whose `data.validation_fraction` is 0.
- Every generated config is then checked like `compute validate` would, including the dataset headers. Nothing is written if any is invalid; variants failing for the same reason are reported once. Warnings are printed once per variant.
- Generating again rewrites only the configs whose text changed, so unchanged runs keep their artifacts, and removes the run directories that the study no longer generates, artifacts included.

## Data preparation

`compute train` loads the datasets before training:

1. Both CSV files are read whole. The header names the columns; every value must be a finite number.
2. Classification: labels must be integers. The classes are the distinct labels of the whole training file, sorted, so labels `1, 2, 3` map to network outputs `0, 1, 2` and MNIST labels `0..=9` map to themselves. A test label missing from the training file is an error. Targets become one-hot vectors.
3. `train_subset` samples are drawn from the training file, then `validation_fraction` of them is held out for validation (rounded to the nearest sample count). For classification both draws are stratified: every class keeps its share of samples. Both use a random stream derived from `training.seed`, separate from other random choices, so the split does not change when e.g. the network changes.
4. Normalization statistics are computed per column on the remaining training samples only, then applied to the training, validation and test sets. `standard` subtracts the mean and divides by the standard deviation, `min_max` maps the training range to `[0, 1]`. A constant column is only shifted to 0.

A `train_subset` larger than the training set, or a `validation_fraction` that leaves the validation or training set empty, is reported at its line in `config.toml`.

## Network

The network (`network::Mlp`) is a stack of fully connected layers stored as matrices, one row per sample:

- Layer sizes are `features -> hidden_layers... -> outputs`, where outputs are the number of classes (one-hot targets) or 1 for regression. Without hidden layers the network is a single layer.
- A layer computes `Z = A_prev . W + b` and `A = f(Z)`. `W` is `in x out`, `b` has length `out` and is broadcast over the rows; with `bias = false` no layer has a bias.
- Hidden layers use `hidden_activation`, the last layer `output_activation`. Softmax subtracts the row maximum before `exp`, and sigmoid is evaluated in a form that does not overflow.
- Every layer is initialized with the same `init` scheme: `xavier` draws from `U[-sqrt(6 / (in + out)), sqrt(6 / (in + out))]`, `he` from `N(0, 2 / in)`, `uniform` from `U[-limit, limit]`. Biases start at 0. Weights come from a random stream of `training.seed` reserved for initialization, so they do not change with e.g. the data split.
- The forward pass keeps `Z` and `A` of every layer for backpropagation; prediction keeps only the current activations.
- Losses are batch means: `mse` and `mae` average over all `n x K` entries, `cross_entropy` averages over samples and is computed from the log-softmax of `Z`, so it stays finite for extreme outputs.
- Backpropagation starts from `dL/dZ` of the output layer: `(A - Y) / n` for `cross_entropy` with `softmax`, otherwise `dL/dA` passed through the output activation. Each layer then gives `dW = A_prev^T . dZ`, `db = sum of dZ over samples` and `dA_prev = dZ . W^T`. At kinks (ReLU at 0, MAE at `a = y`) the derivative is 0. Unit tests compare all gradients with central differences.

`compute train` builds the network after preparing the data and prints its parameter count per layer, e.g. `parameters: 371 (2x16+16, 16x16+16, 16x3+3)`.

## Training

`compute train` trains the network with mini-batch gradient descent (`training::train`):

- Every epoch visits the training samples in a new random order, drawn from a random stream of `training.seed` reserved for shuffling, and splits them into batches of `batch_size`. The last batch is smaller when `batch_size` does not divide the number of samples, so an epoch takes `ceil(n / batch_size)` iterations.
- Each iteration runs the forward pass on the batch, computes the batch loss, backpropagates it and lets the optimizer update every weight and bias. With `g` the gradient of a parameter `w`:
  - `sgd`: `w = w - lr * g`.
  - `momentum`: classical (Polyak) momentum, `v = momentum * v - lr * g`, `w = w + v`, with `v` starting at 0. Source: Sutskever et al. (2013), [On the importance of initialization and momentum in deep learning](https://proceedings.mlr.press/v28/sutskever13.html), equation (1); Goodfellow et al., [Deep Learning](https://www.deeplearningbook.org/contents/optimization.html), algorithm 8.2.
  - `adam`: `m = beta1 * m + (1 - beta1) * g`, `v = beta2 * v + (1 - beta2) * g^2`, `w = w - lr * m_hat / (sqrt(v_hat) + epsilon)` with the bias-corrected `m_hat = m / (1 - beta1^t)`, `v_hat = v / (1 - beta2^t)` after `t` steps. Source: Kingma and Ba (2015), [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980), algorithm 1.
- Before training and after every epoch the network is evaluated on the training, validation and test sets: the loss (on normalized targets), and the accuracy (classification) or the MSE and MAE in the original target scale (regression). The test set is only observed; it never influences training.
- Training stops early when the training loss becomes NaN or infinite. The run is reported as diverged with a warning, not as an error.
- With `training.early_stopping`, an epoch improves when its validation loss is lower than the best one so far by more than `min_delta`. Training stops after `patience` epochs in a row without improvement, and the network of the best epoch (possibly epoch 0) is restored, also when all epochs were run or training diverged.

The same seed and config always give the same training run. The command prints the metrics before training, about 20 times during training (`--progress-every N` changes this) and after the last epoch, then a summary:

```text
epoch 200/200  train loss 0.1303 acc 94.93% | test loss 0.1727 acc 94.07%
stop:       completed after 200 epoch(s), 9400 iteration(s) in 0.62 s
result:     train loss 0.1303 acc 94.93% | test loss 0.1727 acc 94.07%
```

## Artifacts

After training, `compute train` writes the run to `<experiment-dir>/artifacts/` (git-ignored) for the presentation layer. Artifacts of a previous run are removed first. Arrays are NumPy `.npy` files (`numpy.load`), little-endian, C order; `u64` and `i64` for integers, `f64` otherwise. Layers are numbered from 0 (first hidden layer) to the output layer, `S` is the number of weight snapshots, `L` the number of layers, `K` the number of network outputs.

```text
artifacts/
├── metadata.json             # code version, start time, training time, full config,
│                             # data, network, how training ended, final metrics
├── epochs.csv                # metrics before training (epoch 0) and after every epoch
├── history/                  # loss during training
│   ├── iteration.npy         # (n,) last iteration of every window
│   ├── epoch.npy             # (n,) epoch of that iteration
│   └── loss.npy              # (n,) mean batch loss of the window
├── weights/                  # evolution of the weights
│   ├── iteration.npy         # (S,) iteration of every snapshot, 0 = initial weights
│   ├── layer_<l>_indices.npy # (m, 2) recorded (row, column) of [W; b]
│   ├── layer_<l>_values.npy  # (S, m) their values
│   ├── weight_norms.npy      # (S, L) Frobenius norm of W
│   └── bias_norms.npy        # (S, L) Euclidean norm of b, NaN without bias
├── model/                    # the trained network
│   ├── layer_<l>_weights.npy # (in, out)
│   └── layer_<l>_bias.npy    # (out,), only with bias
├── predictions/<split>/      # train, validation (if any) and test
│   ├── rows.npy              # (n,) row of every sample in the CSV file (0 = first data row)
│   ├── outputs.npy           # (n, K) class probabilities, or predictions in the original scale
│   ├── labels.npy            # (n,) true class labels (classification)
│   ├── predicted.npy         # (n,) label of the largest output (classification)
│   └── targets.npy           # (n, 1) true targets in the original scale (regression)
└── grid/                     # only for 1 or 2 input features
    ├── x.npy, y.npy          # (r,) grid axes in the original scale (y: 2 features only)
    └── outputs.npy           # (r, K) or (r, r, K); [i, j] is the point (x[j], y[i])
```

- **Loss history:** every batch loss enters the mean of a window of `logging.loss_every` iterations, weighted by the batch size; the last window may be shorter. With `loss_every = 1` the history holds every batch loss.
- **Weights:** the parameters of a layer form the matrix `[W; b]` of shape `(in + 1) x out`, whose last row (`row == in`) is the bias (without bias just `W`). A layer with more than `logging.max_weights_per_layer` parameters is recorded through a fixed random subset of that size, drawn from a random stream of `training.seed` reserved for it. Snapshots are taken before training, every `logging.weights_every` iterations and after the last iteration. Norms always use all weights.
- **Model and predictions:** with early stopping, the network of the best epoch; predictions are made by that network. Inputs are not copied: `rows` points into the CSV files named in the config.
- **Grid:** `output.grid_resolution` points per axis spanning all samples (training, validation and test) in the original scale, widened by `output.grid_margin` of the range on each side.
- **`epochs.csv`:** columns `epoch`, `iteration`, then for every split (`train`, `validation`, `test`) the loss and either `accuracy` or `mse` and `mae` (original scale), e.g. `train_loss`, `test_accuracy`. A diverged run has `NaN` or `inf` values, which become `null` in `metadata.json`.
- **`metadata.json`:** `version` holds the crate version, the git commit the binary was built from and whether the crate sources differed from it (`dirty`); `started_at` is in seconds since the Unix epoch; `result.stop.reason` is `completed`, `early_stopped` or `diverged` (with `epoch` and `iteration`).

The same seed and config give identical artifacts, except for times in `metadata.json`. Recording happens in memory and writing after training, so it hardly slows training down; compare with `--no-artifacts`.

## Reproducibility

Every random choice comes from `training.seed`. Each source of randomness has its own ChaCha8 stream of that seed (`rng::RngStream`), so changing one part of the config does not change the others:

| Stream | Id | Used for |
|---|---|---|
| `Data` | 0 | training subset and validation split |
| `Init` | 1 | initial weights |
| `Shuffle` | 2 | order of the training samples in every epoch |
| `WeightLog` | 3 | subset of the weights recorded in the artifacts |

The ids are part of the reproducibility contract: changing them would stop old seeds from reproducing old runs. No code uses an unseeded generator, and runs trained in parallel do not share state, so results do not depend on `--jobs`. `build.rs` embeds the git commit and whether the crate sources differed from it, and `metadata.json` stores both with the full config.

## Code layout

Modules of the library in `src/`, each a `<name>.rs` file with a `<name>/` folder of submodules (details in their `//!` docs). Unit tests sit next to the code in `#[cfg(test)] mod tests`.

| Module | Content |
|---|---|
| `cli` | `clap` commands `validate`, `train` and `generate`: discovery of experiments, single and parallel (`rayon`) training, progress and summaries |
| `config` | serde schema of `config.toml` (one file per section), validation rules, loading with diagnostics at the config line; `examples.rs` trains the example and showcase configs end to end |
| `study` | `study.toml`: axes, validation at its lines, expansion into variants and seeds, writing `runs/`; `examples.rs` checks that every committed study generates valid configs (the dataset files of MNIST, which is downloaded, only when present) |
| `data` | CSV reading, class encoding, training subset and validation split, scalers fitted on the training set |
| `network` | `Layer`, `Mlp` (forward pass with cache, prediction, backpropagation), activations, losses, initialization; gradient checks against central differences |
| `training` | the training loop, optimizers, batch order, metrics, early stopping and the `TrainingObserver` trait |
| `artifacts` | `Recorder` (an observer buffering history in memory) and writing of `.npy`, `epochs.csv` and `metadata.json` |
| `rng` | seeded random streams (see [Reproducibility](#reproducibility)) |

Numbers are `f64` and matrices `ndarray::Array2<f64>` with one row per sample. User-facing errors use `thiserror` and `miette`: all problems are reported at once, each at its source location with a hint.

## Commands

```bash
cargo build --release                        # build
cargo run --release -- --help                # show the CLI help
cargo test                                   # run all tests
cargo test <test_name>                       # run a single test
cargo clippy --all-targets -- -D warnings    # lint (as in CI)
cargo fmt                                    # format (CI: cargo fmt --check)
```
