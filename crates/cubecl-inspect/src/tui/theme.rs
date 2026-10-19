//! The browser's styles, on the [palette](crate::palette), and the widgets
//! both screens draw in them. Foreground only:
//! the terminal keeps its own background and its own body text colour, so the
//! browser reads on a light terminal as on a dark one.

use crate::palette::{self, Rgb};
use crate::wordmark;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, HighlightSpacing, List, ListItem, Padding, Paragraph};

pub(crate) const MUTED: Color = terminal(palette::MUTED);
pub(crate) const RULE: Color = terminal(palette::RULE);
pub(crate) const ACCENT: Color = terminal(palette::ACCENT);

/// A colour of the palette, as the terminal library takes it.
const fn terminal(Rgb(red, green, blue): Rgb) -> Color {
    Color::Rgb(red, green, blue)
}

/// A label, the way the site sets its mono captions.
pub(crate) fn label() -> Style {
    Style::new().fg(MUTED)
}

/// The accent, as the site sets an accented heading line.
pub(crate) fn accent() -> Style {
    Style::new().fg(ACCENT).add_modifier(Modifier::BOLD)
}

/// A rule: borders and dividers.
pub(crate) fn rule() -> Style {
    Style::new().fg(RULE)
}

/// The selected row of a list: in the accent, the terminal's background left
/// as it is.
pub(crate) fn selected() -> Style {
    accent()
}

/// A bordered panel, titled the way the site labels a section.
pub(crate) fn panel(title: &str) -> Block<'static> {
    Block::bordered()
        .border_style(rule())
        .title(Span::styled(format!(" {} ", title.to_uppercase()), label()))
        .padding(Padding::horizontal(1))
}

/// A list whose selection is marked by an accent bar, the rows aligned
/// whether selected or not.
pub(crate) fn selectable(items: Vec<ListItem<'_>>) -> List<'_> {
    List::new(items)
        .highlight_style(selected())
        .highlight_symbol(Span::styled("▌ ", Style::new().fg(ACCENT)))
        .highlight_spacing(HighlightSpacing::Always)
}

/// The status bar: what happened last, then each key and what it does.
pub(crate) fn hints<'a>(status: &str, keys: &[(&'a str, &'a str)]) -> Line<'a> {
    let mut spans = vec![Span::raw(format!(" {status}"))];
    for (key, action) in keys {
        spans.push(Span::raw("   "));
        spans.push(Span::styled(
            *key,
            Style::new().add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::styled(format!(" {action}"), label()));
    }
    Line::from(spans)
}

/// The `CubeCL` wordmark, the mark in the accent, indented as the captions
/// under it are.
pub(crate) fn wordmark() -> Paragraph<'static> {
    let art: Vec<Line> = wordmark::MARK
        .iter()
        .zip(wordmark::LETTERS)
        .map(|(mark, letters)| {
            Line::from(vec![
                Span::raw("  "),
                Span::styled(*mark, Style::new().fg(ACCENT)),
                Span::raw(letters),
            ])
        })
        .collect();
    Paragraph::new(art)
}

/// What an environment's screen opens with: the wordmark over its caption
/// when the screen has room for both beside a page worth reading, a single
/// line when it does not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Masthead {
    Wordmark,
    Line,
}

impl Masthead {
    /// The rows a screen keeps for its page before it gives up the wordmark.
    const PAGE_ROWS: u16 = 24;

    /// The masthead a screen of `area` has room for.
    pub(crate) fn fitting(area: Rect) -> Self {
        let wide = area.width as usize >= Self::wordmark_width();
        let tall = area.height >= Self::Wordmark.height() + Self::PAGE_ROWS;
        match wide && tall {
            true => Self::Wordmark,
            false => Self::Line,
        }
    }

    /// The rows it takes: the wordmark between blank rows, then its caption.
    pub(crate) fn height(self) -> u16 {
        match self {
            Self::Wordmark => wordmark::MARK.len() as u16 + 3,
            Self::Line => 1,
        }
    }

    /// The columns the wordmark takes, its indent included.
    fn wordmark_width() -> usize {
        2 + wordmark::MARK[0].chars().count() + wordmark::LETTERS[0].chars().count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_wordmark_shows_only_beside_a_page_worth_reading() {
        let screen = |width, height| Masthead::fitting(Rect::new(0, 0, width, height));
        assert_eq!(screen(170, 50), Masthead::Wordmark);
        assert_eq!(screen(170, 30), Masthead::Line, "too short for both");
        assert_eq!(screen(60, 50), Masthead::Line, "too narrow for the letters");
    }
}
