use crate::palette;
use clap::builder::styling::{Color, Style};
use std::fmt;

/// What a piece of a report is, for a terminal to colour.
///
/// A report rendered with the alternate flag, `format!("{:#}", Text(report))`,
/// paints each role in the [palette](crate::palette)'s colours; the terminal
/// browser reads that same painted form. Rendered plainly, the text carries no
/// escape codes, which is what a pipe and a file read. Everything else keeps
/// the terminal's own colour, so the output reads on a light background too.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Role {
    /// A table's column names, muted and bold.
    Header,
    /// The rule under a table's column names.
    Rule,
    /// The name of a field before its value, muted.
    Label,
    /// A section's title, in the accent.
    Title,
    /// A line that carries the answer, such as a table's total, in bold.
    Strong,
    /// A line that holds nothing, such as a pool with no memory, muted.
    Idle,
    /// What a report says of how to read it, under a table: muted.
    Note,
    /// A measured time, in italics, so a column of them reads apart from the
    /// counts beside it.
    Measure,
}

impl Role {
    fn style(self) -> Style {
        let muted = Style::new().fg_color(Some(Color::from(palette::MUTED)));
        match self {
            Self::Header => muted.bold(),
            Self::Rule => Style::new().fg_color(Some(Color::from(palette::RULE))),
            Self::Label | Self::Idle | Self::Note => muted,
            Self::Title => Style::new()
                .fg_color(Some(Color::from(palette::ACCENT)))
                .bold(),
            Self::Strong => Style::new().bold(),
            Self::Measure => Style::new().italic(),
        }
    }

    /// Write `text` in this role: painted when `f` renders the alternate form.
    pub(crate) fn write(self, f: &mut fmt::Formatter<'_>, text: impl fmt::Display) -> fmt::Result {
        Paint(&[self]).write(f, text)
    }
}

/// Roles laid over one another on a piece of text, as a table paints a cell
/// of a painted row in a painted column: their effects add up, and the first
/// role that has a colour sets it.
pub(crate) struct Paint<'a>(pub &'a [Role]);

impl Paint<'_> {
    fn style(&self) -> Style {
        self.0.iter().fold(Style::new(), |style, role| {
            let own = role.style();
            style
                .effects(style.get_effects() | own.get_effects())
                .fg_color(style.get_fg_color().or(own.get_fg_color()))
        })
    }

    /// Write `text` in these roles: painted when `f` renders the alternate
    /// form and there is a role to paint.
    pub(crate) fn write(&self, f: &mut fmt::Formatter<'_>, text: impl fmt::Display) -> fmt::Result {
        if f.alternate() && !self.0.is_empty() {
            let style = self.style();
            write!(f, "{style}{text}{style:#}")
        } else {
            write!(f, "{text}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Labelled;

    impl fmt::Display for Labelled {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            Role::Label.write(f, "id")
        }
    }

    struct Layered;

    impl fmt::Display for Layered {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            Paint(&[Role::Strong, Role::Measure, Role::Label]).write(f, "9 ms")
        }
    }

    #[test]
    fn layered_roles_add_their_effects_and_keep_a_colour() {
        assert_eq!(format!("{Layered}"), "9 ms");
        assert_eq!(
            format!("{Layered:#}"),
            "\x1b[1m\x1b[3m\x1b[38;2;137;141;149m9 ms\x1b[0m"
        );
    }

    #[test]
    fn only_the_alternate_form_is_painted() {
        assert_eq!(format!("{Labelled}"), "id");
        assert_eq!(format!("{Labelled:#}"), "\x1b[38;2;137;141;149mid\x1b[0m");
    }
}
