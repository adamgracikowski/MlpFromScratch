//! Checks of a parsed study that do not need the generated configs.
//!
//! The configs of the variants are checked afterwards, when the study is
//! expanded (see [`super::load_study`]).

use std::collections::HashSet;

use super::StudyConfig;
use super::axis::{Axis, is_safe_name};
use crate::config::Issue;

/// Key set by `seeds`, which no axis and not `[base]` may set.
const SEED_KEY: &str = "training.seed";

/// Checks seeds, axis and level names, override keys and report roles.
///
/// Returns every problem found, so that all of them can be reported at once.
pub fn validate(study: &StudyConfig) -> Vec<Issue> {
    let mut issues = Vec::new();
    validate_seeds(&study.seeds, &mut issues);
    if study
        .base
        .get("training")
        .and_then(|training| training.get("seed"))
        .is_some()
    {
        issues.push(
            Issue::error(&["base", "training", "seed"], "is set by `seeds`")
                .with_help("remove `seed` from `[base.training]`"),
        );
    }
    let mut names = HashSet::new();
    for (index, axis) in study.axes.iter().enumerate() {
        if !is_safe_name(&axis.name) {
            issues.push(unsafe_name(index, "name", &axis.name));
        } else if !names.insert(axis.name.as_str()) {
            issues.push(
                axis_issue(index, &format!("duplicate axis `{}`", axis.name)).at_field("name"),
            );
        }
        validate_axis(index, axis, &study.base, &mut issues);
    }
    validate_conflicts(&study.axes, &mut issues);
    validate_report(study, &mut issues);
    issues
}

fn validate_seeds(seeds: &[u64], issues: &mut Vec<Issue>) {
    if seeds.is_empty() {
        issues.push(
            Issue::error(&["seeds"], "must list at least one seed")
                .with_help("e.g. `seeds = [1, 2, 3, 4, 5]`"),
        );
    }
    let mut seen = HashSet::new();
    for (index, seed) in seeds.iter().enumerate() {
        if *seed > i64::MAX as u64 {
            issues.push(
                Issue::error(
                    &["seeds"],
                    format!("seed {seed} does not fit a TOML integer"),
                )
                .at_index(index),
            );
        }
        if !seen.insert(seed) {
            issues.push(Issue::error(&["seeds"], format!("duplicate seed {seed}")).at_index(index));
        }
    }
}

fn validate_axis(index: usize, axis: &Axis, base: &toml::Table, issues: &mut Vec<Issue>) {
    let level_names: Vec<String> = match (&axis.key, &axis.values, &axis.levels) {
        (Some(key), Some(values), None) => {
            if let Err(message) = check_key(key, base) {
                issues.push(axis_issue(index, &message).at_field("key"));
            }
            if values.is_empty() {
                issues.push(axis_issue(index, "must list at least one value").at_field("values"));
            }
            let names = axis.levels().into_iter().map(|level| level.name);
            for (position, name) in names.clone().enumerate() {
                if !is_safe_name(&name) {
                    issues.push(
                        axis_issue(
                            index,
                            &format!("level name `{name}` derived from the value is not a valid directory name"),
                        )
                        .at_field("values")
                        .at_index(position)
                        .with_help("name the levels explicitly: `levels = [{ name = \"...\", set = { ... } }]`"),
                    );
                }
            }
            names.collect()
        }
        (None, None, Some(levels)) => {
            if levels.is_empty() {
                issues.push(axis_issue(index, "must list at least one level").at_field("levels"));
            }
            for (position, level) in levels.iter().enumerate() {
                let issue = |message: &str| {
                    axis_issue(index, message)
                        .at_field("levels")
                        .at_index(position)
                };
                if !is_safe_name(&level.name) {
                    issues.push(
                        issue(&format!("`{}` is not a valid directory name", level.name))
                            .at_field("name")
                            .with_help(NAME_HELP),
                    );
                }
                if level.set.is_empty() {
                    issues.push(issue("must set at least one key").at_field("set"));
                }
                for key in level.set.keys() {
                    if let Err(message) = check_key(key, base) {
                        issues.push(
                            issue(&message).at_field("set").with_help(
                                "quote dotted keys: `set = { \"data.train\" = \"...\" }`",
                            ),
                        );
                    }
                }
            }
            levels.iter().map(|level| level.name.clone()).collect()
        }
        _ => {
            issues.push(
                axis_issue(index, "needs either `key` and `values`, or `levels`").with_help(
                    "shorthand: `key = \"model.hidden_activation\"`, `values = [...]`; \
                     full: `levels = [{ name = \"...\", set = { \"data.train\" = \"...\" } }]`",
                ),
            );
            Vec::new()
        }
    };

    let mut seen = HashSet::new();
    for (position, name) in level_names.iter().enumerate() {
        if !seen.insert(name) {
            let field = if axis.levels.is_some() {
                "levels"
            } else {
                "values"
            };
            issues.push(
                axis_issue(index, &format!("duplicate level `{name}`"))
                    .at_field(field)
                    .at_index(position),
            );
        }
    }
}

const NAME_HELP: &str = "use letters, digits, `.`, `_` and `-`";

/// Checks that `key` is a dotted path whose parent table exists in `base`
/// and that it is not the seed key.
fn check_key(key: &str, base: &toml::Table) -> Result<(), String> {
    let segments: Vec<&str> = key.split('.').collect();
    let valid_segment = |segment: &&str| {
        !segment.is_empty()
            && segment
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_')
    };
    if !segments.iter().all(valid_segment) {
        return Err(format!("`{key}` is not a dotted config key"));
    }
    if key == SEED_KEY {
        return Err(format!("`{SEED_KEY}` is set by `seeds`"));
    }
    let (_, parents) = segments
        .split_last()
        .expect("split yields at least one segment");
    let mut table = base;
    for (depth, segment) in parents.iter().enumerate() {
        match table.get(*segment).and_then(toml::Value::as_table) {
            Some(child) => table = child,
            None => {
                let parent = segments[..=depth].join(".");
                return Err(format!("`{key}` needs the table `[base.{parent}]`"));
            }
        }
    }
    Ok(())
}

/// Reports axes that set the same key, or a key and a key inside it.
fn validate_conflicts(axes: &[Axis], issues: &mut Vec<Issue>) {
    let keys: Vec<Vec<String>> = axes.iter().map(Axis::keys).collect();
    for later in 1..axes.len() {
        for earlier in 0..later {
            let conflict = keys[later].iter().find(|key| {
                keys[earlier]
                    .iter()
                    .any(|other| overlaps(key.as_str(), other.as_str()))
            });
            if let Some(key) = conflict {
                issues.push(
                    axis_issue(
                        later,
                        &format!(
                            "sets `{key}`, which axis `{}` also sets",
                            axes[earlier].name
                        ),
                    )
                    .with_help("merge the two axes into one with the `levels` form"),
                );
            }
        }
    }
}

/// Whether setting `a` and `b` touches the same value.
fn overlaps(a: &str, b: &str) -> bool {
    let inside = |inner: &str, outer: &str| {
        inner == outer || (inner.starts_with(outer) && inner[outer.len()..].starts_with('.'))
    };
    inside(a, b) || inside(b, a)
}

fn validate_report(study: &StudyConfig, issues: &mut Vec<Issue>) {
    let axes: HashSet<&str> = study.axes.iter().map(|axis| axis.name.as_str()).collect();
    let report = &study.report;
    let roles = report
        .x
        .iter()
        .map(|name| (Issue::error(&["report", "x"], ""), name))
        .chain(
            report
                .color
                .iter()
                .map(|name| (Issue::error(&["report", "color"], ""), name)),
        )
        .chain(report.panels.iter().enumerate().map(|(index, name)| {
            (
                Issue::error(&["report", "panels"], "").at_index(index),
                name,
            )
        }));
    let mut assigned = HashSet::new();
    for (issue, name) in roles {
        let message = if !axes.contains(name.as_str()) {
            format!("`{name}` is not an axis of the study")
        } else if !assigned.insert(name.as_str()) {
            format!("axis `{name}` already has a role")
        } else {
            continue;
        };
        issues.push(Issue { message, ..issue });
    }
}

fn axis_issue(index: usize, message: &str) -> Issue {
    Issue::error(&["axes"], message).at_index(index)
}

fn unsafe_name(index: usize, field: &'static str, name: &str) -> Issue {
    axis_issue(index, &format!("`{name}` is not a valid directory name"))
        .at_field(field)
        .with_help(NAME_HELP)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::study::fixtures::STUDY;

    fn errors(text: &str) -> Vec<String> {
        let study: StudyConfig = toml::from_str(text).unwrap();
        validate(&study).iter().map(Issue::to_string).collect()
    }

    #[test]
    fn valid_study_has_no_issues() {
        assert_eq!(errors(STUDY), Vec::<String>::new());
    }

    #[test]
    fn rejects_bad_seeds() {
        assert_eq!(
            errors(&STUDY.replace("seeds = [1, 2]", "seeds = []")),
            ["`seeds`: must list at least one seed"]
        );
        assert_eq!(
            errors(&STUDY.replace("seeds = [1, 2]", "seeds = [1, 2, 1]")),
            ["`seeds[2]`: duplicate seed 1"]
        );
    }

    #[test]
    fn rejects_seed_in_base_and_axes() {
        let text = STUDY
            .replace("batch_size = 4", "batch_size = 4\nseed = 3")
            .replace("\"model.hidden_layers\"", "\"training.seed\"");
        assert_eq!(
            errors(&text),
            [
                "`base.training.seed`: is set by `seeds`",
                "`axes[1].key`: `training.seed` is set by `seeds`",
            ]
        );
    }

    #[test]
    fn rejects_bad_keys() {
        let text = STUDY
            .replace("\"model.hidden_activation\"", "\"model..activation\"")
            .replace("\"model.hidden_layers\"", "\"network.layers\"");
        assert_eq!(
            errors(&text),
            [
                "`axes[0].key`: `model..activation` is not a dotted config key",
                "`axes[1].key`: `network.layers` needs the table `[base.network]`",
            ]
        );
    }

    #[test]
    fn rejects_bad_and_duplicate_names() {
        let text = STUDY
            .replace("name = \"layers\"", "name = \"activation\"")
            .replace("{ kind = \"tanh\" }", "{ kind = \"sigmoid\" }")
            .replace("[[], [8, 8]]", "[\"a/b\"]");
        assert_eq!(
            errors(&text),
            [
                "`axes[0].values[1]`: duplicate level `sigmoid`",
                "`axes[1].name`: duplicate axis `activation`",
                "`axes[1].values[0]`: level name `a/b` derived from the value is not a valid directory name",
                "`report.panels[0]`: `layers` is not an axis of the study",
            ]
        );
    }

    #[test]
    fn checks_full_form_levels() {
        let text = STUDY.replace(
            "key = \"model.hidden_layers\"\nvalues = [[], [8, 8]]",
            "levels = [\n  { name = \"small\", set = { \"data.train_subset\" = 2 } },\n  { name = \"bad name\", set = {} },\n  { name = \"x\", set = { \"data.split.part\" = 1 } },\n]",
        );
        assert_eq!(
            errors(&text),
            [
                "`axes[1].levels[1].name`: `bad name` is not a valid directory name",
                "`axes[1].levels[1].set`: must set at least one key",
                "`axes[1].levels[2].set`: `data.split.part` needs the table `[base.data.split]`",
            ]
        );
    }

    #[test]
    fn rejects_axis_without_one_form() {
        let text = STUDY.replace("values = [[], [8, 8]]", "");
        assert_eq!(
            errors(&text),
            ["`axes[1]`: needs either `key` and `values`, or `levels`"]
        );
    }

    #[test]
    fn rejects_axes_setting_the_same_value() {
        let text = STUDY
            .replace("\"model.hidden_layers\"", "\"model\"")
            .replace("[[], [8, 8]]", "[{ bias = true }]");
        assert_eq!(
            errors(&text),
            ["`axes[1]`: sets `model`, which axis `activation` also sets"]
        );
        assert!(overlaps("model.bias", "model.bias"));
        assert!(!overlaps("model.bias", "model.bias_init"));
    }

    #[test]
    fn rejects_bad_report_roles() {
        let text = STUDY.replace(
            "panels = [\"layers\"]",
            "color = \"depth\"\npanels = [\"activation\"]",
        );
        assert_eq!(
            errors(&text),
            [
                "`report.color`: `depth` is not an axis of the study",
                "`report.panels[0]`: axis `activation` already has a role",
            ]
        );
    }
}
