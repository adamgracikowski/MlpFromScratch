# Presentation Layer

Python package that reads the training artifacts written by the compute layer (`src/compute/`) and turns them into plots, tables and Markdown summaries. It does no training. See the [main README](../../README.md) for an overview of the project.

The code lives in the `analysis` package (see [Code layout](#code-layout)). `main.py` is the entry point. Python >= 3.13 and the dependencies (`numpy`, `matplotlib`, `polars`; dev: `pytest`, `ruff`) are managed with `uv` and locked in `uv.lock`.

## Usage

Run from the repository root, because dataset paths in the experiment configs are relative to the current working directory:

```bash
cargo run --release --manifest-path src/compute/Cargo.toml -- train data/experiments/classification/example-three-gauss
uv run --project src/analysis src/analysis/main.py plot data/experiments/classification/example-three-gauss
```

`plot <experiment-dir>` reads `<experiment-dir>/artifacts/` (written by `compute train`) and replaces `<experiment-dir>/plots/` with the plots and `summary.md`. Options:

- `--format pdf,png` (default): comma-separated file formats out of `pdf`, `png`, `svg`.
- `--dpi 200` (default): resolution of raster images.

`aggregate <study-dir>` aggregates a study (see `src/compute/README.md`, section Studies) over seeds. It reads `<study-dir>/study.toml`, `runs/index.csv` (written by `compute generate`) and the `metadata.json` and `epochs.csv` of every run, then replaces `<study-dir>/results/` with:

- `runs.csv`: one row per run: directory, variant, seed, level of every axis, status (`completed`, `early_stopped`, `diverged` or `missing` without artifacts), epochs, best epoch, training time and the final metrics of every split (accuracy as a fraction). Regression runs also get `test_mse_inside` and `test_mse_outside`: the test MSE (original scale) over the test samples whose inputs lie in the range (minimum to maximum, per feature) of the run's training inputs, and over the others. The test sets of the regression datasets reach beyond the training inputs, so the test MSE mixes interpolation with extrapolation. They are computed from `predictions/` of the runs.
- `variants.csv`: one row per variant: numbers of runs, finished, diverged and missing runs, and the mean, sample standard deviation, minimum, first quartile, median, third quartile and maximum of every metric over the finished runs (completed or early stopped).
- the comparison plots and galleries below and `summary.md` (plan, base config, results table, representative runs).

Options: `--format` and `--dpi` as for `plot`; `--plot-runs` also runs `plot` for the representative run of every variant: the finished run whose test metric is closest to the median over seeds (the smaller seed on a tie). Runs without artifacts are listed as a warning; a study without any trained run is an error.

```bash
cargo run --release --manifest-path src/compute/Cargo.toml -- generate data/experiments/classification/<study>
cargo run --release --manifest-path src/compute/Cargo.toml -- train data/experiments/classification/<study>
uv run --project src/analysis src/analysis/main.py aggregate data/experiments/classification/<study> --plot-runs
```

The comparison plots compare the test accuracy (classification) or the test MSE in the original scale (regression): losses of different loss functions are not comparable. `[report]` of `study.toml` places the axes: one panel per combination of the `panels` levels, the `x` levels as categories and the `color` levels as colors; by default the last axis is `x` and the others are panels. The learning curves put the `color` levels into panels too. `labels` of `[report]` (axis or level name -> text, e.g. Polish names) replace the names on the plots and in the results table of `summary.md`.

| File | Content | When |
|---|---|---|
| `seeds` | final test metric: mean with +-1 standard deviation (accuracy) or median with the quartiles (errors) and a dot per seed; diverged runs noted | always |
| `curves` | test metric per epoch: mean over seeds with a +-1 standard deviation band (accuracy) or median with a band between the quartiles (errors); a panel per combination of the `panels` and `color` levels (levels of `color` often differ in the number of epochs), a curve per `x` level; the accuracy axis fits the curves without the first 10 % of epochs (the climb from chance level), the band is clipped to 0-100 % | always |
| `curves_train_loss` | training loss per epoch, as `curves` (median and quartiles, log axis) | all variants use the same loss |
| `heatmap` | mean +- standard deviation (accuracy) or median (quartiles) (errors) of the test metric, rows `color`, columns `x` | `x` and `color` are set |
| `seeds_inside`, `seeds_outside`, `heatmap_inside`, `heatmap_outside` | `seeds` and `heatmap` of `test_mse_inside` (interpolation) and `test_mse_outside` (extrapolation) | regression (`heatmap_*`: as `heatmap`) |
| `gallery[_<level>__<level>]` | one figure per combination of the levels of the axes other than `x`, a panel per `x` level (three columns, four panels as 2 x 2; decision regions 6 cm per column): decision regions (classification) or network curve (regression) of the representative run, with its test metric; at most 600 test (classification) or training (regression) samples per panel | classification with 2 features, regression with 1 |

Statistics use the finished runs only. Diverged runs are counted separately, and so are runs without artifacts. Errors (log axes) are summarized by the median and the quartiles: over seeds they have a long tail, a few bad seeds pull the mean up, and mean - standard deviation is often negative. The results table of `summary.md` uses the same statistics.

## Plots

Texts on the plots and in `summary.md` are in Polish, like the report. Figures are drawn at their final print size (16 cm wide, 10 pt STIX font), with a decimal comma, the Okabe-Ito palette and no titles. Files are reproducible: the same artifacts give byte-identical files.

| File | Content | When |
|---|---|---|
| `dataset` | training, validation and test samples | classification with 2 features, regression with 1 |
| `decision_boundary` | class regions shaded by confidence, boundaries, test samples (at most 3000, evenly spaced), misclassified ones | classification with 2 features |
| `regression` | data, target function (test set) and network output; test residuals | regression with 1 feature |
| `loss` | loss per iteration window with a moving average; loss per epoch for every split | always |
| `metrics` | accuracy (classification) or MSE in the original scale (regression) per epoch | always |
| `weights_trajectories` | values of the sampled weights and biases, one panel per layer | always |
| `weight_norms` | Frobenius norm of `W` and Euclidean norm of `b` of every layer | always |
| `weights_heatmap` | sampled parameters (sorted by final value) over the snapshots | always |
| `confusion_<split>` | confusion matrix with counts and row percentages | classification |
| `summary.md` | configuration, course of training, final metrics, code version | always |

Plots that do not fit the data (e.g. the decision boundary for MNIST) are skipped with a message.

## Code layout

| Path | Content |
|---|---|
| `main.py`, `analysis/cli.py` | entry point and the `argparse` commands `plot` and `aggregate` |
| `analysis/experiment/` | artifacts of one run: `load_experiment(dir) -> Experiment`, one model per file (`LossHistory`, `WeightHistory`, predictions per split, `Grid`), `ArtifactsError` for missing or broken files; inputs are read from the config's CSV files through `rows.npy` |
| `analysis/study/` | `study.toml` and `runs/index.csv` of a study, the summary of every run (`metadata.json` and `epochs.csv` only), results over seeds (`variants_table`, representative runs) and the regression errors inside and outside the training range |
| `analysis/plot_experiment.py` | the list of experiment plots (name, drawing function, when it applies, why it is skipped); writes `plots/` and `summary.md` |
| `analysis/aggregate_study.py` | the list of study plots, galleries and tables; writes `results/` |
| `analysis/summary.py`, `analysis/study_summary.py` | `summary.md` of an experiment and of a study |
| `analysis/plots/` | one module per kind of plot, each `plot_*(...) -> Figure`; `study_panels.py` holds what the study plots share (panels, mean +- sd or median and quartiles) |
| `analysis/style.py` | rcParams, figure sizes in cm and deterministic saving |
| `analysis/formatting.py` | decimal comma on ticks and in tables |
| `analysis/labels.py` | every Polish text of the plots and summaries; the only module with non-ASCII characters |
| `analysis/series.py` | gaps after divergence, moving average, evenly spaced subsampling |
| `tests/` | `pytest` tests mirroring the package; `conftest.py` provides `make_experiment` and `make_study`, which write small synthetic runs and studies |

## Commands

```bash
uv sync                       # install dependencies
uv run main.py --help         # show the CLI help
uv run pytest                 # run tests
uv run ruff check             # lint
uv run ruff format            # format (CI: ruff format --check)
```

If `uv` fails with `invalid peer certificate: UnknownIssuer` (for example behind a TLS-intercepting proxy or antivirus), add `--system-certs` to use the system certificate store.
