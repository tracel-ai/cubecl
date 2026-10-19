//! One environment open in the browser: its reports, a tab each, read again
//! whenever the file changes.

use super::base::shown;
use super::theme::{self, Masthead, hints, panel, selectable};
use crate::report::{AutotuneReport, CandidateReport, KeyOrder};
use crate::view::{KeyText, Maybe, Text, Wall};
use crate::wordmark;
use crate::{InspectError, Inspector};
use ansi_to_tui::IntoText as _;
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, ListItem, ListState, Padding, Paragraph, Tabs};
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// One environment open in the browser: its reports as last read, which tab
/// shows, where it is scrolled, and which autotune key is selected or open.
pub(crate) struct Viewer {
    path: PathBuf,
    tab: Tab,
    scroll: u16,
    pages: Option<Pages>,
    keys: ListState,
    /// The key opened from the autotune list, rendered.
    detail: Option<Page>,
    /// What the file looked like when last read.
    seen: Option<Fingerprint>,
}

/// The reports, rendered for their tabs; the autotune one kept whole, since
/// its keys are selectable.
struct Pages {
    /// What the manifest calls the environment, when it says more than the
    /// file's name.
    name: Option<String>,
    summary: Page,
    timeline: Page,
    autotune: AutotuneReport,
    /// The autotune keys as rows, in the report's order: formatted once a
    /// reading rather than on every frame the list is drawn.
    keys: Vec<KeyRow>,
    candidates: Page,
    kernels: Page,
    memory: Page,
}

/// A report as a tab shows it: painted the way a terminal paints it, once a
/// reading rather than on every frame.
struct Page(ratatui::text::Text<'static>);

/// One autotune key's row: its id, set apart, then its tune's wall, its
/// tuner padded to the widest, and its winner; under them, the key itself,
/// which is what tells two rows of one tuner apart.
struct KeyRow {
    id: String,
    rest: String,
    key: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tab {
    Summary,
    Timeline,
    Autotune,
    Candidates,
    Kernels,
    Memory,
}

/// The file and its write-ahead log, by size and modification time: a build
/// writes to the log first, so the log is what moves while it runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Fingerprint([(u64, Option<SystemTime>); 2]);

impl Tab {
    const ALL: [Tab; 6] = [
        Tab::Summary,
        Tab::Timeline,
        Tab::Autotune,
        Tab::Candidates,
        Tab::Kernels,
        Tab::Memory,
    ];

    fn title(self) -> &'static str {
        match self {
            Tab::Summary => "summary",
            Tab::Timeline => "timeline",
            Tab::Autotune => "autotune",
            Tab::Candidates => "candidates",
            Tab::Kernels => "kernels",
            Tab::Memory => "memory",
        }
    }

    fn index(self) -> usize {
        Self::ALL.iter().position(|tab| *tab == self).unwrap_or(0)
    }

    fn step(self, by: isize) -> Self {
        let count = Self::ALL.len() as isize;
        Self::ALL[(self.index() as isize + by).rem_euclid(count) as usize]
    }
}

impl Fingerprint {
    fn of(path: &Path) -> Self {
        let stat = |path: &Path| {
            std::fs::metadata(path).map_or((0, None), |meta| (meta.len(), meta.modified().ok()))
        };
        let mut wal = path.as_os_str().to_owned();
        wal.push("-wal");
        Self([stat(path), stat(Path::new(&wal))])
    }
}

impl KeyRow {
    /// Every key of `report`, a row each, the tuner column padded to its
    /// widest.
    fn all(report: &AutotuneReport) -> Vec<Self> {
        let tuners = report
            .keys
            .iter()
            .map(|key| key.table.tuner.chars().count())
            .max()
            .unwrap_or_default();
        report
            .keys
            .iter()
            .map(|key| Self {
                id: key.id.to_string(),
                rest: format!(
                    "  {:>9}  {:<tuners$}  {}",
                    Maybe(key.wall().map(Wall)).to_string(),
                    key.table.tuner,
                    key.winner_name()
                ),
                key: KeyText(&key.key).to_string(),
            })
            .collect()
    }

    /// The row as the list draws it, borrowing its text: two lines, the key
    /// indented under the id.
    fn item(&self) -> ListItem<'_> {
        let indent = " ".repeat(self.id.chars().count() + 2);
        ListItem::new(vec![
            Line::from(vec![
                Span::styled(self.id.as_str(), theme::label()),
                Span::raw(self.rest.as_str()),
            ]),
            Line::from(vec![
                Span::raw(indent),
                Span::styled(self.key.as_str(), theme::label()),
            ]),
        ])
    }
}

impl Pages {
    fn read(inspector: &Inspector) -> Self {
        let mut autotune = inspector.autotune();
        autotune.sort(KeyOrder::Wall);
        let summary = inspector.summary();
        Self {
            name: summary.name().map(str::to_string),
            summary: Page::of(Text(&summary)),
            timeline: Page::of(Text(&inspector.timeline())),
            candidates: Page::of(Text(&CandidateReport::from(&autotune))),
            keys: KeyRow::all(&autotune),
            autotune,
            kernels: Page::of(Text(&inspector.kernels())),
            memory: Page::of(Text(&inspector.memory())),
        }
    }

    /// The page a tab shows; `None` for the autotune tab, drawn as a list.
    fn page(&self, tab: Tab) -> Option<&Page> {
        match tab {
            Tab::Summary => Some(&self.summary),
            Tab::Timeline => Some(&self.timeline),
            Tab::Candidates => Some(&self.candidates),
            Tab::Kernels => Some(&self.kernels),
            Tab::Memory => Some(&self.memory),
            Tab::Autotune => None,
        }
    }
}

impl Page {
    /// `report` in its painted form. Painting writes only the escape codes
    /// the roles use, so the conversion cannot fail on them; were it to,
    /// the page falls back to the plain form rather than showing nothing.
    fn of(report: impl fmt::Display) -> Self {
        let painted = format!("{report:#}");
        Self(
            painted
                .as_bytes()
                .into_text()
                .unwrap_or_else(|_| ratatui::text::Text::raw(report.to_string())),
        )
    }

    /// The page, scrolled.
    fn scrolled(&self, scroll: u16) -> Paragraph<'_> {
        Paragraph::new(self.0.clone()).scroll((scroll, 0))
    }
}

impl Viewer {
    pub(crate) fn new(path: PathBuf) -> Self {
        Self {
            path,
            tab: Tab::Summary,
            scroll: 0,
            pages: None,
            keys: ListState::default().with_selected(Some(0)),
            detail: None,
            seen: None,
        }
    }

    /// Read every report again. A file mid-write that cannot be read keeps
    /// the previous reading on screen; the error says why.
    pub(crate) fn read(&mut self) -> Result<(), InspectError> {
        self.seen = Some(Fingerprint::of(&self.path));
        self.pages = Some(Pages::read(&Inspector::open(&self.path)?));
        Ok(())
    }

    /// Whether the file moved since it was last read.
    pub(crate) fn changed(&self) -> bool {
        self.seen != Some(Fingerprint::of(&self.path))
    }

    /// The path of the environment open: what a test of the browser's
    /// transitions asks of it.
    #[cfg(test)]
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Move to the tab `by` steps along, wrapping at either end.
    pub(crate) fn step_tab(&mut self, by: isize) {
        self.switch(self.tab.step(by));
    }

    fn switch(&mut self, tab: Tab) {
        self.tab = tab;
        self.scroll = 0;
        self.detail = None;
    }

    /// Scroll a page, or move the selection in the key list.
    pub(crate) fn move_by(&mut self, rows: isize) {
        let selecting = self.tab == Tab::Autotune && self.detail.is_none();
        if selecting {
            let count = self
                .pages
                .as_ref()
                .map_or(0, |pages| pages.autotune.keys.len());
            let selected = self.keys.selected().unwrap_or(0) as isize + rows;
            self.keys.select(Some(
                selected.clamp(0, count.saturating_sub(1) as isize) as usize
            ));
        } else {
            self.scroll = (self.scroll as isize + rows).max(0) as u16;
        }
    }

    pub(crate) fn open_key(&mut self) {
        if self.tab != Tab::Autotune || self.detail.is_some() {
            return;
        }
        let key = self
            .pages
            .as_ref()
            .zip(self.keys.selected())
            .and_then(|(pages, index)| pages.autotune.keys.get(index));
        if let Some(key) = key {
            self.detail = Some(Page::of(Text(key)));
            self.scroll = 0;
        }
    }

    /// Close the key open, answering whether there was one: with none, a
    /// close backs out of the environment.
    pub(crate) fn close_key(&mut self) -> bool {
        self.scroll = 0;
        self.detail.take().is_some()
    }

    pub(crate) fn draw(&mut self, frame: &mut Frame, status: &str) {
        let masthead = Masthead::fitting(frame.area());
        let [header, tabs, body, footer] = Layout::vertical([
            Constraint::Length(masthead.height()),
            Constraint::Length(2),
            Constraint::Fill(1),
            Constraint::Length(1),
        ])
        .areas(frame.area());

        let built_for = self.pages.as_ref().and_then(|pages| pages.name.as_deref());
        let environment = [
            Some(Span::styled(
                name(&self.path),
                Style::new().add_modifier(Modifier::BOLD),
            )),
            built_for.map(|built_for| Span::raw(format!("  {built_for}"))),
            Some(Span::styled(
                format!("  {}", shown(&self.path)),
                theme::label(),
            )),
        ]
        .into_iter()
        .flatten();
        match masthead {
            Masthead::Wordmark => {
                let [_, brand, _, caption] = Layout::vertical([
                    Constraint::Length(1),
                    Constraint::Length(wordmark::MARK.len() as u16),
                    Constraint::Length(1),
                    Constraint::Length(1),
                ])
                .areas(header);
                frame.render_widget(theme::wordmark(), brand);
                let lead = [
                    Span::raw("  "),
                    Span::styled("ENVIRONMENT INSPECTOR", theme::accent()),
                    Span::styled("  ·  ", theme::label()),
                ];
                frame.render_widget(
                    Line::from_iter(lead.into_iter().chain(environment)),
                    caption,
                );
            }
            Masthead::Line => {
                let lead = [
                    Span::styled(" CubeCL", theme::accent()),
                    Span::styled(" inspect  ", theme::label()),
                ];
                frame.render_widget(Line::from_iter(lead.into_iter().chain(environment)), header);
            }
        }
        frame.render_widget(
            Tabs::new(Tab::ALL.map(Tab::title))
                .select(self.tab.index())
                .style(theme::label())
                .highlight_style(theme::accent().add_modifier(Modifier::UNDERLINED))
                .divider(Span::styled("·", theme::rule()))
                .padding(" ", " ")
                .block(Block::new().padding(Padding::top(1))),
            tabs,
        );

        let block = panel(self.tab.title());
        match (&self.pages, self.tab, &self.detail) {
            (None, _, _) => frame.render_widget(
                Paragraph::new("reading…")
                    .style(theme::label())
                    .block(block),
                body,
            ),
            (Some(_), Tab::Autotune, Some(detail)) => {
                frame.render_widget(detail.scrolled(self.scroll).block(block), body)
            }
            (Some(pages), Tab::Autotune, None) => frame.render_stateful_widget(
                selectable(pages.keys.iter().map(KeyRow::item).collect()).block(block),
                body,
                &mut self.keys,
            ),
            (Some(pages), tab, _) => {
                if let Some(page) = pages.page(tab) {
                    frame.render_widget(page.scrolled(self.scroll).block(block), body)
                }
            }
        }

        let back = match self.detail {
            Some(_) => "keys",
            None => "environments",
        };
        frame.render_widget(
            hints(
                status,
                &[
                    ("←→", "tab"),
                    ("↑↓", "move"),
                    ("⏎", "open"),
                    ("esc", back),
                    ("r", "read"),
                    ("q", "quit"),
                ],
            ),
            footer,
        );
    }
}

/// An environment's file name, which is its name.
fn name(path: &Path) -> String {
    path.file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tabs_wrap_both_ways() {
        assert_eq!(Tab::Summary.step(-1), Tab::Memory);
        assert_eq!(Tab::Memory.step(1), Tab::Summary);
        assert_eq!(Tab::Timeline.step(1), Tab::Autotune);
    }

    /// The selection stays on the list, whatever the key asks.
    #[test]
    fn the_selection_is_clamped_to_the_keys() {
        let mut viewer = Viewer::new(PathBuf::from("missing.cubecl"));
        viewer.switch(Tab::Autotune);
        viewer.move_by(-5);
        assert_eq!(viewer.keys.selected(), Some(0));
        viewer.move_by(5);
        assert_eq!(viewer.keys.selected(), Some(0), "no keys read");
    }

    /// The browser shows a report as a terminal paints it, not plainly.
    #[test]
    fn a_page_keeps_the_report_s_painting() {
        struct Titled;

        impl fmt::Display for Titled {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                crate::view::Role::Title.write(f, "tuned")
            }
        }

        let Page(text) = Page::of(Titled);
        let span = &text.lines[0].spans[0];
        assert_eq!(span.content, "tuned");
        assert!(span.style.add_modifier.contains(Modifier::BOLD));
        assert_eq!(span.style.fg, Some(theme::ACCENT));
    }
}
