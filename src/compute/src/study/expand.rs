//! Expansion of a study into one config per variant and seed.

use std::path::PathBuf;

use toml::Value;

use super::{BASE_VARIANT, LEVEL_SEPARATOR, Level, RUNS_DIR, StudyConfig};

/// One run of a study before its config is checked.
#[derive(Debug, Clone, PartialEq)]
pub struct RunSpec {
    /// Directory of the run relative to the study directory,
    /// `runs/<variant>/seed-<seed>`.
    pub dir: PathBuf,
    /// Level names joined by `__`, or `base` for a study without axes.
    pub variant: String,
    /// Name of the level of every axis, in axis order.
    pub levels: Vec<String>,
    /// Value of `training.seed`.
    pub seed: u64,
    /// `[base]` with the overrides of the levels and the seed.
    pub config: toml::Table,
}

/// Lists every combination of levels (the first axis varies slowest) and,
/// within each, every seed in the order of `seeds`.
///
/// Expects a study without validation errors: override keys have parent
/// tables in `[base]` and every axis has levels.
pub fn expand(study: &StudyConfig) -> Vec<RunSpec> {
    let axes: Vec<Vec<Level>> = study.axes.iter().map(|axis| axis.levels()).collect();
    let mut runs = Vec::new();
    for combination in combinations(&axes) {
        let mut config = study.base.clone();
        for level in &combination {
            for (key, value) in &level.set {
                set(&mut config, key, value.clone());
            }
        }
        let levels: Vec<String> = combination.iter().map(|level| level.name.clone()).collect();
        let variant = if levels.is_empty() {
            BASE_VARIANT.to_owned()
        } else {
            levels.join(LEVEL_SEPARATOR)
        };
        for &seed in &study.seeds {
            let mut config = config.clone();
            let seed_value = i64::try_from(seed).expect("validated seeds fit a TOML integer");
            set(&mut config, "training.seed", Value::Integer(seed_value));
            runs.push(RunSpec {
                dir: [RUNS_DIR, &variant, &format!("seed-{seed}")]
                    .iter()
                    .collect(),
                variant: variant.clone(),
                levels: levels.clone(),
                seed,
                config,
            });
        }
    }
    runs
}

/// Cartesian product of the levels of all axes; one empty combination for
/// no axes.
fn combinations(axes: &[Vec<Level>]) -> Vec<Vec<&Level>> {
    let mut combinations: Vec<Vec<&Level>> = vec![Vec::new()];
    for levels in axes {
        combinations = combinations
            .into_iter()
            .flat_map(|prefix| {
                levels.iter().map(move |level| {
                    let mut combination = prefix.clone();
                    combination.push(level);
                    combination
                })
            })
            .collect();
    }
    combinations
}

/// Replaces the value at the dotted `key`, creating missing parent tables
/// (validation already reported missing parents of axis keys; the seed key
/// creates `[training]` only in an invalid base).
fn set(table: &mut toml::Table, key: &str, value: Value) {
    let mut segments: Vec<&str> = key.split('.').collect();
    let last = segments.pop().expect("split yields at least one segment");
    let mut current = table;
    for segment in segments {
        let entry = current
            .entry(segment)
            .or_insert_with(|| Value::Table(toml::Table::new()));
        if !entry.is_table() {
            *entry = Value::Table(toml::Table::new());
        }
        current = entry.as_table_mut().expect("entry was made a table");
    }
    current.insert(last.to_owned(), value);
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::study::fixtures::STUDY;

    fn study(text: &str) -> StudyConfig {
        toml::from_str(text).unwrap()
    }

    #[test]
    fn expands_product_of_levels_and_seeds() {
        let runs = expand(&study(STUDY));
        let dirs: Vec<&Path> = runs.iter().map(|run| run.dir.as_path()).collect();
        let expected: Vec<PathBuf> = [
            "runs/sigmoid__0/seed-1",
            "runs/sigmoid__0/seed-2",
            "runs/sigmoid__8-8/seed-1",
            "runs/sigmoid__8-8/seed-2",
            "runs/tanh__0/seed-1",
            "runs/tanh__0/seed-2",
            "runs/tanh__8-8/seed-1",
            "runs/tanh__8-8/seed-2",
        ]
        .iter()
        .map(PathBuf::from)
        .collect();
        assert_eq!(dirs, expected);
        assert_eq!(runs[3].levels, ["sigmoid", "8-8"]);
        assert_eq!(runs[3].variant, "sigmoid__8-8");
        assert_eq!(runs[3].seed, 2);
    }

    #[test]
    fn applies_overrides_and_seed() {
        let run = &expand(&study(STUDY))[6];
        let model = run.config["model"].as_table().unwrap();
        assert_eq!(model["hidden_activation"]["kind"].as_str(), Some("tanh"));
        assert_eq!(model["hidden_layers"].as_array().unwrap().len(), 2);
        assert_eq!(model["bias"].as_bool(), Some(true));
        assert_eq!(run.config["training"]["seed"].as_integer(), Some(1));
    }

    #[test]
    fn replaces_whole_value_of_a_key() {
        let text = STUDY.replace(
            "values = [{ kind = \"sigmoid\" }, { kind = \"tanh\" }]",
            "values = [{ kind = \"leaky_relu\", alpha = 0.1 }, { kind = \"tanh\" }]",
        );
        let runs = expand(&study(&text));
        let activation = |run: &RunSpec| run.config["model"]["hidden_activation"].clone();
        assert_eq!(activation(&runs[0])["alpha"].as_float(), Some(0.1));
        assert!(activation(&runs[4]).get("alpha").is_none());
    }

    #[test]
    fn study_without_axes_has_base_variant() {
        let mut study = study(STUDY);
        study.axes.clear();
        let runs = expand(&study);
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].dir, PathBuf::from("runs/base/seed-1"));
        assert!(runs[0].levels.is_empty());
    }
}
