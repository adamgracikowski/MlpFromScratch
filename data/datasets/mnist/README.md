# MNIST

Handwritten digits 0-9: 60000 training and 10000 test images of 28 x 28 grayscale pixels (0-255). The dataset is not in the repository; this directory holds it locally (everything except this README is git-ignored).

## Preparing the dataset

Run from the repository root:

```bash
uv run --project src/analysis src/analysis/main.py mnist
```

The command downloads the four gzipped IDX files from the public mirror `https://storage.googleapis.com/cvdf-datasets/mnist/` (no login needed), checks the SHA-256 of every uncompressed file and converts them to CSV. Running it again only verifies the files; `--force` rewrites the CSV files.

The project description points to the copy on [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset/data), which needs a login. To use it, download the archive and pass it (or the directory it was unpacked to):

```bash
uv run --project src/analysis src/analysis/main.py mnist --source path/to/archive.zip
```

Both sources hold the same original files, so the checksums match either way.

## Files

| File | Size | Content |
|---|---|---|
| `raw/train-images-idx3-ubyte` | 47 040 016 B | training images, IDX (`uint8`, 60000 x 28 x 28) |
| `raw/train-labels-idx1-ubyte` | 60 008 B | training labels, IDX (`uint8`, 60000) |
| `raw/t10k-images-idx3-ubyte` | 7 840 016 B | test images, IDX (`uint8`, 10000 x 28 x 28) |
| `raw/t10k-labels-idx1-ubyte` | 10 008 B | test labels, IDX (`uint8`, 10000) |
| `data.mnist.train.60000.csv` | about 105 MB | training set for the compute layer |
| `data.mnist.test.10000.csv` | about 17 MB | test set for the compute layer |

The CSV files have a header row, the target column `label` (0-9) and the raw pixel values (0-255) in columns `pixel0` to `pixel783`, row by row of the image (`pixel<28 * row + column>`). A config reads them with `target = "label"`; the scaling of the pixels is set by `data.input_normalization`. In memory the compute layer holds the training set as 60000 x 784 `f64` values, about 376 MB.

Training uses only the training file (`data.train_subset` and `data.validation_fraction` draw from it); the test file is only for evaluating the trained network.
