//! A valid study and temporary study directories shared by the study tests.

use std::fs;

use tempfile::TempDir;

use super::STUDY_FILE_NAME;

/// A valid study: two activations x two architectures, two seeds.
pub const STUDY: &str = r#"
description = "Activation and depth"
seeds = [1, 2]

[[axes]]
name = "activation"
key = "model.hidden_activation"
values = [{ kind = "sigmoid" }, { kind = "tanh" }]

[[axes]]
name = "layers"
key = "model.hidden_layers"
values = [[], [8, 8]]

[report]
x = "activation"
panels = ["layers"]

[base]
task = "classification"

[base.data]
train = "train.csv"
test = "test.csv"
target = "cls"
input_normalization = "standard"
target_normalization = "none"
validation_fraction = 0.0

[base.model]
hidden_layers = [16]
bias = true
hidden_activation = { kind = "relu" }
output_activation = "softmax"
init = { kind = "xavier" }

[base.optimizer]
kind = "adam"
learning_rate = 0.01

[base.training]
loss = "cross_entropy"
epochs = 5
batch_size = 4

[base.logging]
loss_every = 1
weights_every = 10
max_weights_per_layer = 8

[base.output]
grid_resolution = 10
grid_margin = 0.1
"#;

/// A study directory with `study.toml` made from `text`, whose dataset
/// paths point at temporary CSV files.
pub fn study_dir(text: &str) -> TempDir {
    let dir = TempDir::new().unwrap();
    let train = dir.path().join("train.csv");
    let test = dir.path().join("test.csv");
    fs::write(
        &train,
        "x,y,cls\n0.1,0.2,1\n0.5,0.1,2\n0.3,0.9,1\n0.8,0.7,2\n",
    )
    .unwrap();
    fs::write(&test, "x,y,cls\n0.3,0.4,2\n0.6,0.2,1\n").unwrap();
    let text = text
        .replace("\"train.csv\"", &format!("'{}'", train.display()))
        .replace("\"test.csv\"", &format!("'{}'", test.display()));
    fs::write(dir.path().join(STUDY_FILE_NAME), text).unwrap();
    dir
}
