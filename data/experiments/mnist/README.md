# MNIST experiments

Part two of the project: the network of part one recognizes the handwritten digits of MNIST. Nothing was added to the network or the training; only the configuration changes. Training uses the 60000 training images only. The 10000 test images evaluate the final networks and influence no choice.

## Data

The dataset is not in the repository. Download it and convert it to CSV (see [`data/datasets/mnist/README.md`](../../datasets/mnist/README.md)):

```bash
uv run --project src/analysis src/analysis/main.py mnist
```

## Tuning

Five studies, one factor at a time, each starting from the choices of the previous ones. Every variant trains on a stratified random subset of 10000 training images, of which 2000 are held out for validation, with 5 seeds. The variants are compared on the validation accuracy (`split = "validation"` in `[report]` of `study.toml`), with early stopping on the validation loss (patience 5, at most 30 epochs).

| Study | Levels | Chosen | Validation accuracy |
|---|---|---|---|
| [`tuning-normalization`](tuning-normalization/results/summary.md) | min-max, standardization (per pixel) | min-max | 94.35 % +- 0.61 (standardization 93.90 % +- 0.18) |
| [`tuning-architectures`](tuning-architectures/results/summary.md) | hidden layers: none, 32, 128, 512, 128-64, 256-128, 128-128-128 | 512 | 95.16 % +- 0.39 (256-128: 94.71 % +- 1.11) |
| [`tuning-activations`](tuning-activations/results/summary.md) | sigmoid, tanh (Xavier initialization), ReLU, leaky ReLU (He) | ReLU | 95.16 % +- 0.39 (leaky ReLU: 95.12 % +- 0.48) |
| [`tuning-optimizers`](tuning-optimizers/results/summary.md) | SGD (0.1), momentum (0.01, 0.9), Adam (0.0003, 0.001, 0.003) | Adam, 0.001 | 95.16 % +- 0.39 (Adam 0.003: 94.71 % +- 0.56) |
| [`tuning-batch-sizes`](tuning-batch-sizes/results/summary.md) | 16, 32, 64, 128 | 32 | 95.16 % +- 0.39 (64: 94.92 % +- 0.58) |

Mean +- sample standard deviation over the 5 seeds; in brackets the runner-up. The base of the first study was the example [`example-mnist`](example-mnist/config.toml) (784-128-10, ReLU, Adam 0.001, batch 32); the base configuration with 512 neurons appears in four studies with the same seeds and gives the same networks in all of them.

## Final configuration

784-512-10, ReLU with He initialization, softmax output with cross-entropy, min-max scaling of every pixel, Adam with learning rate 0.001, batch size 32. It trains on all 60000 training images, of which 6000 are held out for early stopping (patience 5, at most 50 epochs).

| Experiment | Seeds | Test accuracy |
|---|---|---|
| [`final-seeds`](final-seeds/results/summary.md) | 1, 2, 3, 4, 5 | 97.91 % +- 0.15 (97.74 %, 98.12 %, 97.95 %, 97.80 %, 97.92 %) |
| [`final`](final/plots/summary.md) | 1 | 97.74 % |

`final` is the submitted network. Its seed was fixed in advance, not chosen by the test result; it is the same run as seed 1 of `final-seeds`. It stopped after 10 epochs and kept the weights of epoch 5 (training accuracy 99.34 %, validation 97.73 %). Its plots in [`final/plots/`](final/plots/) include the confusion matrices, the most confident mistakes on the test set (`misclassified`) and the weights of the first layer as images (`first_layer_weights`).

## Reproducing the submitted network

Everything that determines the training is in [`final/config.toml`](final/config.toml), including the seed (`training.seed = 1`). The network was trained with the code of commit `8034b38`, which [`final/artifacts/metadata.json`](final/artifacts/metadata.json) also records. Training is deterministic: the same config, seed and code give the same weights, bit for bit. Run from the repository root:

```bash
uv run --project src/analysis src/analysis/main.py mnist
cargo run --release --manifest-path src/compute/Cargo.toml -- train data/experiments/mnist/final
uv run --project src/analysis src/analysis/main.py plot data/experiments/mnist/final
```

`train` replaces `final/artifacts/`. The committed parameters of the network (`final/artifacts/model/*.npy`) and `metadata.json` can be restored with `git checkout` to compare, e.g. with `git diff --stat -- data/experiments/mnist/final/artifacts/model`.

The studies are reproduced with `scripts/run-studies.ps1` (PowerShell 7), which generates, trains and aggregates them one after another:

```powershell
pwsh -File scripts/run-studies.ps1 -Jobs 1 data/experiments/mnist/final-seeds data/experiments/mnist/tuning-normalization data/experiments/mnist/tuning-architectures data/experiments/mnist/tuning-activations data/experiments/mnist/tuning-optimizers data/experiments/mnist/tuning-batch-sizes
```

With `-Jobs 1` the 125 runs take about 40 minutes on a laptop.
