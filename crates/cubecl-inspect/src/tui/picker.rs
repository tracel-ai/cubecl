//! The environments to choose from, under the wordmark.

use super::base::shown;
use super::theme::{self, hints, panel, selectable};
use crate::report::Listing;
use crate::view::Table;
use crate::{EnvironmentDirectory, wordmark};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout};
use ratatui::text::{Line, Span};
use ratatui::widgets::{ListItem, ListState, Paragraph};
use std::path::{Path, PathBuf};

/// The environments to choose from, as last listed.
pub(crate) struct Picker {
    /// The listing table's column names.
    header: String,
    choices: Vec<Choice>,
    selection: ListState,
    /// Files found that could not be opened.
    unreadable: usize,
}

/// One environment in the list: its file, and its row of the listing table.
struct Choice {
    path: PathBuf,
    row: String,
}

impl Picker {
    /// The environments of `listing`, a row each, in its order.
    pub(crate) fn new(listing: &Listing) -> Self {
        let mut lines = Table::from(listing.environments.as_slice()).lines();
        let header = lines.remove(0);
        // `Table::from` adds one row per environment, in their order, so the
        // rows and the environments pair up one to one.
        let choices = listing
            .environments
            .iter()
            .zip(lines)
            .map(|(summary, row)| Choice {
                path: summary.path.clone(),
                row,
            })
            .collect();
        Self {
            header,
            choices,
            selection: ListState::default().with_selected(Some(0)),
            unreadable: listing.unreadable.len(),
        }
    }

    /// The file of the environment selected, when there is one.
    pub(crate) fn selected(&self) -> Option<&Path> {
        self.selection
            .selected()
            .and_then(|index| self.choices.get(index))
            .map(|choice| choice.path.as_path())
    }

    /// Move the selection, kept on the list.
    pub(crate) fn move_by(&mut self, rows: isize) {
        let selected = self.selection.selected().unwrap_or(0) as isize + rows;
        let last = self.choices.len().saturating_sub(1) as isize;
        self.selection
            .select(Some(selected.clamp(0, last) as usize));
    }

    pub(crate) fn status(&self) -> String {
        match self.unreadable {
            0 => format!("{} environments", self.choices.len()),
            unreadable => format!(
                "{} environments, {unreadable} unreadable",
                self.choices.len()
            ),
        }
    }

    /// The list under the wordmark, with the directories it was read from.
    pub(crate) fn draw(
        &mut self,
        frame: &mut Frame,
        directories: &[EnvironmentDirectory],
        status: &str,
    ) {
        let [_, brand, _, caption, _, body, footer] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(wordmark::MARK.len() as u16),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Fill(1),
            Constraint::Length(1),
        ])
        .areas(frame.area());

        frame.render_widget(theme::wordmark(), brand);

        let searched: Vec<String> = directories
            .iter()
            .map(|directory| shown(&directory.path))
            .collect();
        frame.render_widget(
            Line::from(vec![
                Span::raw("  "),
                Span::styled("ENVIRONMENT INSPECTOR", theme::accent()),
                Span::styled(format!("  ·  {}", searched.join("  ·  ")), theme::label()),
            ]),
            caption,
        );

        let block = panel("environments");
        if self.choices.is_empty() {
            frame.render_widget(
                Paragraph::new("no environment here yet: a build writes one")
                    .style(theme::label())
                    .block(block),
                body,
            );
        } else {
            let inner = block.inner(body);
            frame.render_widget(block, body);
            let [header, rows] =
                Layout::vertical([Constraint::Length(1), Constraint::Fill(1)]).areas(inner);
            frame.render_widget(
                Line::styled(format!("  {}", self.header), theme::label()),
                header,
            );
            let items: Vec<ListItem> = self
                .choices
                .iter()
                .map(|choice| ListItem::new(choice.row.as_str()))
                .collect();
            frame.render_stateful_widget(selectable(items), rows, &mut self.selection);
        }

        frame.render_widget(
            hints(
                status,
                &[
                    ("↑↓", "move"),
                    ("⏎", "open"),
                    ("r", "list again"),
                    ("q", "quit"),
                ],
            ),
            footer,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::Summary;

    fn summary(path: &str) -> Summary {
        Summary {
            path: PathBuf::from(path),
            file_bytes: 0,
            manifest: None,
            namespaces: Vec::new(),
            sessions: Vec::new(),
        }
    }

    /// Each row of the list is the environment it opens: the rows come from
    /// the listing table built one row per environment, in its order.
    #[test]
    fn each_row_pairs_with_its_environment() {
        let picker = Picker::new(&Listing {
            environments: vec![summary("first.db"), summary("second.db")],
            ..Listing::default()
        });
        assert!(picker.header.starts_with("environment"));
        let rows: Vec<&str> = picker
            .choices
            .iter()
            .map(|choice| choice.row.as_str())
            .collect();
        assert!(rows[0].starts_with("first"));
        assert!(rows[1].starts_with("second"));
    }

    /// The selection stays on the list, whatever the key asks.
    #[test]
    fn the_selection_stays_on_the_list() {
        let mut picker = Picker::new(&Listing {
            environments: vec![summary("a.db"), summary("b.db")],
            ..Listing::default()
        });
        picker.move_by(5);
        assert_eq!(picker.selected(), Some(Path::new("b.db")));
        picker.move_by(-5);
        assert_eq!(picker.selected(), Some(Path::new("a.db")));
    }
}
