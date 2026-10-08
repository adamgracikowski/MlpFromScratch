//! Problems found while checking a parsed config.

use std::fmt;

use miette::Severity;

/// One segment of the path to a config value, e.g. `model.hidden_layers[2]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Key {
    /// Field of a table.
    Field(&'static str),
    /// Element of an array.
    Index(usize),
}

/// A problem with a config value, located by its key path.
#[derive(Debug, Clone, PartialEq)]
pub struct Issue {
    /// `Error` rejects the config; `Warning` and `Advice` are only reported.
    pub severity: Severity,
    /// Path of the offending value, used to point at it in `config.toml`.
    pub path: Vec<Key>,
    /// What is wrong with the value.
    pub message: String,
    /// How to fix it.
    pub help: Option<String>,
}

impl Issue {
    /// Creates an issue at the field `path` (table names, then the field name).
    pub fn new(severity: Severity, path: &[&'static str], message: impl Into<String>) -> Self {
        Self {
            severity,
            path: path.iter().map(|&name| Key::Field(name)).collect(),
            message: message.into(),
            help: None,
        }
    }

    /// Creates an error at the field `path`.
    pub fn error(path: &[&'static str], message: impl Into<String>) -> Self {
        Self::new(Severity::Error, path, message)
    }

    /// Creates a warning at the field `path`.
    pub fn warning(path: &[&'static str], message: impl Into<String>) -> Self {
        Self::new(Severity::Warning, path, message)
    }

    /// Narrows the path to the element `index` of the array it points at.
    pub fn at_index(mut self, index: usize) -> Self {
        self.path.push(Key::Index(index));
        self
    }

    /// Attaches a hint on how to fix the problem.
    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }

    /// Whether the issue rejects the config.
    pub fn is_error(&self) -> bool {
        self.severity == Severity::Error
    }

    /// Dotted form of the path, e.g. `model.hidden_layers[2]`.
    pub fn path_string(&self) -> String {
        let mut path = String::new();
        for key in &self.path {
            match key {
                Key::Field(name) if path.is_empty() => path.push_str(name),
                Key::Field(name) => {
                    path.push('.');
                    path.push_str(name);
                }
                Key::Index(index) => path.push_str(&format!("[{index}]")),
            }
        }
        path
    }
}

impl fmt::Display for Issue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "`{}`: {}", self.path_string(), self.message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_path_with_index() {
        let issue = Issue::error(&["model", "hidden_layers"], "must be positive").at_index(2);
        assert_eq!(issue.path_string(), "model.hidden_layers[2]");
        assert_eq!(
            issue.to_string(),
            "`model.hidden_layers[2]`: must be positive"
        );
        assert!(issue.is_error());
    }

    #[test]
    fn warning_is_not_error() {
        assert!(!Issue::warning(&["task"], "odd").is_error());
    }
}
