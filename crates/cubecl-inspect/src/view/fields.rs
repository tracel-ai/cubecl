use super::paint::Role;
use std::fmt;

/// The space between a label and its value, past the longest label.
const GAP: usize = 2;

/// Lines of `label value` that describe one thing, the labels muted and
/// padded to one column, so the values line up.
pub(crate) struct Fields {
    width: usize,
}

impl Fields {
    /// Lines whose labels are among `labels`: the column fits the longest.
    pub(crate) fn new(labels: &[&str]) -> Self {
        let longest = labels.iter().map(|label| label.chars().count()).max();
        Self {
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
        let width = self.width;
        Role::Label.write(f, format_args!("{label:<width$}"))?;
        writeln!(f, "{value}")
    }
}
