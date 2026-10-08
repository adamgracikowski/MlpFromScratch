//! Rendering of config issues as `miette` diagnostics pointing into `config.toml`.

use std::fmt;

use miette::{Diagnostic, LabeledSpan, NamedSource, Severity, SourceCode, SourceSpan};
use thiserror::Error;
use toml::Spanned;
use toml::de::{DeTable, DeValue};

use super::{Issue, Key};

/// An [`Issue`] together with the location of the offending value.
#[derive(Debug, Error)]
#[error("{issue}")]
pub struct IssueDiagnostic {
    issue: Issue,
    span: Option<SourceSpan>,
    src: NamedSource<String>,
}

impl IssueDiagnostic {
    /// Locates `issue` in the parsed `document` whose text is `src`.
    pub fn new(
        issue: Issue,
        document: Option<&Spanned<DeTable<'_>>>,
        src: NamedSource<String>,
    ) -> Self {
        let span = document.and_then(|document| find_span(document, &issue.path));
        Self { issue, span, src }
    }

    /// The underlying issue.
    pub fn issue(&self) -> &Issue {
        &self.issue
    }

    /// Byte offset of the offending value, used to order diagnostics.
    pub fn offset(&self) -> Option<usize> {
        self.span.map(|span| span.offset())
    }
}

impl Diagnostic for IssueDiagnostic {
    fn severity(&self) -> Option<Severity> {
        Some(self.issue.severity)
    }

    fn help<'a>(&'a self) -> Option<Box<dyn fmt::Display + 'a>> {
        self.issue
            .help
            .as_ref()
            .map(|help| Box::new(help) as Box<dyn fmt::Display>)
    }

    fn source_code(&self) -> Option<&dyn SourceCode> {
        Some(&self.src)
    }

    fn labels(&self) -> Option<Box<dyn Iterator<Item = LabeledSpan> + '_>> {
        let label = match self.issue.severity {
            Severity::Error => "invalid value",
            Severity::Warning => "suspicious value",
            Severity::Advice => "note",
        };
        self.span.map(|span| {
            Box::new(std::iter::once(LabeledSpan::new_with_span(
                Some(label.to_owned()),
                span,
            ))) as Box<dyn Iterator<Item = LabeledSpan>>
        })
    }
}

/// Finds the span of the value at `path`.
///
/// When the path leads to a missing value (e.g. an omitted optional field),
/// falls back to the deepest existing ancestor.
pub fn find_span(document: &Spanned<DeTable<'_>>, path: &[Key]) -> Option<SourceSpan> {
    let (Key::Field(first), rest) = path.split_first()? else {
        return None;
    };
    let mut current: &Spanned<DeValue<'_>> = document.get_ref().get(*first)?;
    for key in rest {
        let next = match key {
            Key::Field(name) => current.get_ref().get(*name),
            Key::Index(index) => current.get_ref().get(*index),
        };
        match next {
            Some(value) => current = value,
            None => break,
        }
    }
    Some(current.span().into())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT: &str = "task = \"regression\"\n\n[model]\nhidden_layers = [8, 0]\ninit = { kind = \"uniform\", limit = -1.0 }\n";

    fn span_text(path: &[Key]) -> Option<&'static str> {
        let document = DeTable::parse(TEXT).unwrap();
        let span = find_span(&document, path)?;
        Some(&TEXT[span.offset()..span.offset() + span.len()])
    }

    #[test]
    fn finds_top_level_value() {
        assert_eq!(span_text(&[Key::Field("task")]), Some("\"regression\""));
    }

    #[test]
    fn finds_array_element() {
        let path = [
            Key::Field("model"),
            Key::Field("hidden_layers"),
            Key::Index(1),
        ];
        assert_eq!(span_text(&path), Some("0"));
    }

    #[test]
    fn finds_inline_table_field() {
        let path = [Key::Field("model"), Key::Field("init"), Key::Field("limit")];
        assert_eq!(span_text(&path), Some("-1.0"));
    }

    #[test]
    fn falls_back_to_existing_ancestor() {
        let path = [Key::Field("model"), Key::Field("init"), Key::Field("alpha")];
        assert_eq!(
            span_text(&path),
            Some("{ kind = \"uniform\", limit = -1.0 }")
        );
    }

    #[test]
    fn missing_section_has_no_span() {
        assert_eq!(span_text(&[Key::Field("data"), Key::Field("train")]), None);
    }
}
