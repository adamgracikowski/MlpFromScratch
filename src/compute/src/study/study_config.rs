//! Schema of `study.toml`.

use std::collections::BTreeMap;

use serde::Deserialize;
use strum::Display;

use super::Axis;

/// A grid of experiments: a base config, axes of overrides and seeds.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StudyConfig {
    /// What the study investigates; copied into the study summary.
    #[serde(default)]
    pub description: Option<String>,
    /// Every variant runs once per seed, which becomes `training.seed`.
    pub seeds: Vec<u64>,
    /// Axes whose levels are combined into variants (cartesian product).
    #[serde(default)]
    pub axes: Vec<Axis>,
    /// Placement of the axes on the comparison plots.
    #[serde(default)]
    pub report: ReportConfig,
    /// Complete experiment config shared by all runs, without `training.seed`.
    /// Kept as raw TOML: the levels override its keys before it is checked.
    pub base: toml::Table,
}

/// Roles of the axes on the comparison plots, by axis name. Axes without a
/// role are added to the panels; without any roles the last axis is on the
/// x axis. Only the presentation layer uses these roles.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReportConfig {
    /// Axis whose levels are the categories on the x axis.
    #[serde(default)]
    pub x: Option<String>,
    /// Axis whose levels are told apart by color within a panel.
    #[serde(default)]
    pub color: Option<String>,
    /// Axes with one panel per combination of their levels.
    #[serde(default)]
    pub panels: Vec<String>,
    /// Texts shown on the plots instead of axis or level names (which are
    /// directory names, so plain ASCII), keyed by axis or level name.
    #[serde(default)]
    pub labels: BTreeMap<String, String>,
    /// Split whose metric compares the variants.
    #[serde(default)]
    pub split: Split,
}

/// A split of the data whose final metric compares the variants of a study.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Display)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum Split {
    /// The training set.
    Train,
    /// The validation set held out from the training set; every variant
    /// needs `data.validation_fraction > 0`.
    Validation,
    /// The test set (default). Choices made on it leak the test set into
    /// the results, so tuning should compare on the validation set.
    #[default]
    Test,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::study::fixtures::STUDY;

    #[test]
    fn parses_study() {
        let study: StudyConfig = toml::from_str(STUDY).unwrap();
        assert_eq!(study.seeds, [1, 2]);
        assert_eq!(study.axes.len(), 2);
        assert_eq!(study.report.x.as_deref(), Some("activation"));
        assert_eq!(study.report.panels, ["layers"]);
        assert_eq!(study.report.labels["layers"], "Hidden layers");
        assert_eq!(study.base["task"].as_str(), Some("classification"));
    }

    #[test]
    fn report_and_description_are_optional() {
        let text = STUDY
            .replace("description = \"Activation and depth\"", "")
            .replace(
                "[report]\nx = \"activation\"\npanels = [\"layers\"]\n\
                 labels = { layers = \"Hidden layers\" }",
                "",
            );
        let study: StudyConfig = toml::from_str(&text).unwrap();
        assert_eq!(study.description, None);
        assert_eq!(study.report, ReportConfig::default());
    }

    #[test]
    fn split_defaults_to_test() {
        let study: StudyConfig = toml::from_str(STUDY).unwrap();
        assert_eq!(study.report.split, Split::Test);
        let text = STUDY.replace("[report]\n", "[report]\nsplit = \"validation\"\n");
        let study: StudyConfig = toml::from_str(&text).unwrap();
        assert_eq!(study.report.split, Split::Validation);
        assert_eq!(study.report.split.to_string(), "validation");
    }

    #[test]
    fn rejects_unknown_split() {
        let text = STUDY.replace("[report]\n", "[report]\nsplit = \"holdout\"\n");
        assert!(toml::from_str::<StudyConfig>(&text).is_err());
    }

    #[test]
    fn rejects_unknown_field() {
        let text = STUDY.replace("seeds =", "seed =");
        assert!(toml::from_str::<StudyConfig>(&text).is_err());
    }
}
