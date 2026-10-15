//! Axes of a study and the names of their levels.

use serde::Deserialize;
use toml::Value;

/// One axis of a study, in one of two forms:
///
/// - shorthand: `key` and `values`, one level per value, named after it;
/// - full: `levels`, each with a name and a table of dotted keys to set.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Axis {
    /// Name of the axis, e.g. `activation`.
    pub name: String,
    /// Dotted config key set by the shorthand form, e.g. `model.hidden_layers`.
    #[serde(default)]
    pub key: Option<String>,
    /// Values of `key` in the shorthand form.
    #[serde(default)]
    pub values: Option<Vec<Value>>,
    /// Levels of the full form.
    #[serde(default)]
    pub levels: Option<Vec<LevelConfig>>,
}

/// A level in the full form of an axis.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LevelConfig {
    /// Name of the level, used in the run directory names.
    pub name: String,
    /// Values to set, by quoted dotted key, e.g. `"data.train" = "..."`.
    pub set: toml::Table,
}

/// A level of an axis, resolved from either form.
#[derive(Debug, Clone, PartialEq)]
pub struct Level {
    /// Name of the level.
    pub name: String,
    /// Dotted config keys and the values that replace them.
    pub set: Vec<(String, Value)>,
}

impl Axis {
    /// Levels in file order; empty when the axis mixes or lacks both forms
    /// (validation reports that).
    pub fn levels(&self) -> Vec<Level> {
        match (&self.key, &self.values, &self.levels) {
            (Some(key), Some(values), None) => values
                .iter()
                .map(|value| Level {
                    name: level_name(value),
                    set: vec![(key.clone(), value.clone())],
                })
                .collect(),
            (None, None, Some(levels)) => levels
                .iter()
                .map(|level| Level {
                    name: level.name.clone(),
                    set: level
                        .set
                        .iter()
                        .map(|(key, value)| (key.clone(), value.clone()))
                        .collect(),
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Dotted config keys set by any level of the axis.
    pub fn keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self
            .levels()
            .into_iter()
            .flat_map(|level| level.set.into_iter().map(|(key, _)| key))
            .collect();
        keys.sort();
        keys.dedup();
        keys
    }
}

/// Name of a shorthand level derived from its value: strings as they are,
/// numbers and booleans as written, arrays joined by `-` (`[]` is `0`, so
/// `hidden_layers = [16, 16]` is `16-16`), and tables as their `kind`
/// followed by the other values (`leaky_relu-0.01`).
pub fn level_name(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Integer(number) => number.to_string(),
        Value::Float(number) => number.to_string(),
        Value::Boolean(flag) => flag.to_string(),
        Value::Datetime(datetime) => datetime.to_string(),
        Value::Array(items) if items.is_empty() => "0".to_owned(),
        Value::Array(items) => join(items.iter()),
        Value::Table(table) => {
            let kind = table.get("kind");
            let others = table
                .iter()
                .filter(|(key, _)| key.as_str() != "kind")
                .map(|(_, value)| value);
            join(kind.into_iter().chain(others))
        }
    }
}

fn join<'a>(values: impl Iterator<Item = &'a Value>) -> String {
    values.map(level_name).collect::<Vec<_>>().join("-")
}

/// Whether `name` can be used as (part of) a directory name.
pub(super) fn is_safe_name(name: &str) -> bool {
    !name.is_empty()
        && name != "."
        && name != ".."
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(text: &str) -> Value {
        toml::from_str::<toml::Table>(&format!("v = {text}")).unwrap()["v"].clone()
    }

    #[test]
    fn names_levels_after_values() {
        assert_eq!(level_name(&value("\"standard\"")), "standard");
        assert_eq!(level_name(&value("32")), "32");
        assert_eq!(level_name(&value("0.01")), "0.01");
        assert_eq!(level_name(&value("false")), "false");
        assert_eq!(level_name(&value("[]")), "0");
        assert_eq!(level_name(&value("[64, 16]")), "64-16");
        assert_eq!(level_name(&value("{ kind = \"tanh\" }")), "tanh");
        assert_eq!(
            level_name(&value("{ kind = \"leaky_relu\", alpha = 0.01 }")),
            "leaky_relu-0.01"
        );
    }

    #[test]
    fn checks_safe_names() {
        assert!(is_safe_name("leaky_relu-0.01"));
        assert!(!is_safe_name(""));
        assert!(!is_safe_name(".."));
        assert!(!is_safe_name("data/simple.csv"));
        assert!(!is_safe_name("a b"));
    }

    fn axis(text: &str) -> Axis {
        toml::from_str(text).unwrap()
    }

    #[test]
    fn resolves_shorthand_levels() {
        let axis = axis("name = \"bias\"\nkey = \"model.bias\"\nvalues = [true, false]");
        let levels = axis.levels();
        assert_eq!(levels.len(), 2);
        assert_eq!(levels[1].name, "false");
        assert_eq!(
            levels[1].set,
            [("model.bias".to_owned(), Value::Boolean(false))]
        );
        assert_eq!(axis.keys(), ["model.bias"]);
    }

    #[test]
    fn resolves_full_levels() {
        let axis = axis(
            "name = \"dataset\"\nlevels = [\n  { name = \"small\", set = { \"data.train\" = \"a.csv\", \"data.test\" = \"b.csv\" } },\n  { name = \"big\", set = { \"data.train\" = \"c.csv\" } },\n]",
        );
        let levels = axis.levels();
        assert_eq!(levels[0].name, "small");
        assert_eq!(levels[0].set.len(), 2);
        assert_eq!(axis.keys(), ["data.test", "data.train"]);
    }

    #[test]
    fn mixed_forms_have_no_levels() {
        let axis = axis(
            "name = \"x\"\nkey = \"model.bias\"\nvalues = [true]\nlevels = [{ name = \"a\", set = {} }]",
        );
        assert!(axis.levels().is_empty());
    }
}
