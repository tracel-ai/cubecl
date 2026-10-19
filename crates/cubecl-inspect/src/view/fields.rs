use super::paint::Role;
use std::fmt;

/// The space between a label and its value, past the longest label.
const GAP: usize = 2;

/// Lines of `label value` that describe one thing, the labels muted and
/// padded to one column, so the values line up.
pub(crate) struct Fields<'a> {
    labels: &'a [&'a str],
    width: usize,
}

impl<'a> Fields<'a> {
    /// Lines whose labels are among `labels`: the column fits the longest.
    pub(crate) fn new(labels: &'a [&'a str]) -> Self {
        let longest = labels.iter().map(|label| label.chars().count()).max();
        Self {
            labels,
            width: longest.unwrap_or_default() + GAP,
        }
    }

    /// Write one line: `label`, padded, then `value`.
    pub(crate) fn line(
        &self,
        f: &mut fmt::Formatter<'_>,
        label: &str,
        value: impl fmt::Display,
    ) -> fmt::Result {
        // A label the column was not sized for would push its value out of
        // line with the others.
        debug_assert!(self.labels.contains(&label), "no field labelled {label}");
        let width = self.width;
        Role::Label.write(f, format_args!("{label:<width$}"))?;
        writeln!(f, "{value}")
    }
}
