# Compute Layer

Rust crate that trains the multilayer perceptron and writes training artifacts. It is split into a library (`src/lib.rs`) and a thin `compute` binary (`src/main.rs`) that provides the CLI.

The Rust version is pinned in `rust-toolchain.toml`.

## Usage

Every command takes an experiment directory that contains `config.toml`:

```bash
compute validate <experiment-dir>   # check the config and dataset files, print a summary
compute train <experiment-dir>      # prepare the data, train the network, print progress and results
compute train <experiment-dir> --progress-every 10   # print metrics every 10 epochs (0: only the summary)
```

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

## Commands

```bash
cargo build --release                        # build
cargo run --release -- --help                # show the CLI help
cargo test                                   # run all tests
cargo test <test_name>                       # run a single test
cargo clippy --all-targets -- -D warnings    # lint (as in CI)
cargo fmt                                    # format
```
