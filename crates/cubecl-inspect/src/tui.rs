//! The reports in a terminal browser: one tab per report, the autotune keys
//! selectable, and everything read again whenever the file changes — open it
//! beside a build and watch the build fill it.

use crate::report::{AutotuneReport, CandidateReport, KeyOrder};
use crate::view::{Maybe, Text, Wall};
use crate::{InspectError, Inspector};
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Modifier, Style};
use ratatui::text::Line;
use ratatui::widgets::{Block, List, ListItem, ListState, Paragraph, Tabs};
use ratatui::{DefaultTerminal, Frame};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

/// How long the loop waits for a key before looking at the file again.
const POLL: Duration = Duration::from_millis(250);

/// How often the file is looked at for a change.
const REFRESH: Duration = Duration::from_secs(1);

/// Rows a page key moves.
const PAGE: u16 = 20;

/// Browse the environment at `path` until the user quits.
pub fn run(path: &Path) -> Result<(), InspectError> {
    let mut explorer = Explorer::new(path.to_path_buf());
    explorer.refresh();
    let mut terminal = ratatui::init();
    let result = explorer.run(&mut terminal);
    ratatui::restore();
    result
}

/// The browser's state: which tab, where it is scrolled, and the reports as
/// last read.
struct Explorer {
    path: PathBuf,
    tab: Tab,
    scroll: u16,
    pages: Option<Pages>,
    keys: ListState,
    /// The key opened from the autotune list, rendered.
    detail: Option<String>,
    /// What the file looked like when last read.
    seen: Option<Fingerprint>,
    checked: Instant,
    status: String,
}

/// The reports, rendered for their tabs; the autotune one kept whole, since
/// its keys are selectable.
struct Pages {
    summary: String,
    timeline: String,
    autotune: AutotuneReport,
    candidates: String,
    kernels: String,
    memory: String,
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

/// What a key press asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Action {
    Quit,
    NextTab,
    PreviousTab,
    Up(u16),
    Down(u16),
    Open,
    Close,
    Refresh,
    Nothing,
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

impl From<KeyEvent> for Action {
    fn from(key: KeyEvent) -> Self {
        if key.kind != KeyEventKind::Press {
            return Action::Nothing;
        }
        match key.code {
            KeyCode::Char('q') => Action::Quit,
            KeyCode::Tab | KeyCode::Right => Action::NextTab,
            KeyCode::BackTab | KeyCode::Left => Action::PreviousTab,
            KeyCode::Up | KeyCode::Char('k') => Action::Up(1),
            KeyCode::Down | KeyCode::Char('j') => Action::Down(1),
            KeyCode::PageUp => Action::Up(PAGE),
            KeyCode::PageDown | KeyCode::Char(' ') => Action::Down(PAGE),
            KeyCode::Enter => Action::Open,
            KeyCode::Esc | KeyCode::Backspace => Action::Close,
            KeyCode::Char('r') => Action::Refresh,
            _ => Action::Nothing,
        }
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

impl Pages {
    fn read(inspector: &Inspector) -> Self {
        let mut autotune = inspector.autotune();
        autotune.sort(KeyOrder::Wall);
        Self {
            summary: Text(&inspector.summary()).to_string(),
            timeline: Text(&inspector.timeline()).to_string(),
            candidates: Text(&CandidateReport::from(&autotune)).to_string(),
            autotune,
            kernels: Text(&inspector.kernels()).to_string(),
            memory: Text(&inspector.memory()).to_string(),
        }
    }

    fn text(&self, tab: Tab) -> &str {
        match tab {
            Tab::Summary => &self.summary,
            Tab::Timeline => &self.timeline,
            Tab::Candidates => &self.candidates,
            Tab::Kernels => &self.kernels,
            Tab::Memory => &self.memory,
            // Drawn as a list, not a page.
            Tab::Autotune => "",
        }
    }
}

impl Explorer {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            tab: Tab::Summary,
            scroll: 0,
            pages: None,
            keys: ListState::default().with_selected(Some(0)),
            detail: None,
            seen: None,
            checked: Instant::now(),
            status: String::new(),
        }
    }

    fn run(&mut self, terminal: &mut DefaultTerminal) -> Result<(), InspectError> {
        loop {
            terminal.draw(|frame| self.draw(frame))?;
            if event::poll(POLL)?
                && let Event::Key(key) = event::read()?
            {
                match Action::from(key) {
                    Action::Quit => return Ok(()),
                    action => self.apply(action),
                }
            }
            if self.checked.elapsed() >= REFRESH {
                self.refresh_if_changed();
            }
        }
    }

    fn apply(&mut self, action: Action) {
        match action {
            Action::NextTab => self.switch(self.tab.step(1)),
            Action::PreviousTab => self.switch(self.tab.step(-1)),
            Action::Up(rows) => self.move_by(-(rows as isize)),
            Action::Down(rows) => self.move_by(rows as isize),
            Action::Open => self.open_key(),
            Action::Close => {
                self.detail = None;
                self.scroll = 0;
            }
            Action::Refresh => self.refresh(),
            Action::Quit | Action::Nothing => {}
        }
    }

    fn switch(&mut self, tab: Tab) {
        self.tab = tab;
        self.scroll = 0;
        self.detail = None;
    }

    /// Scroll a page, or move the selection in the key list.
    fn move_by(&mut self, rows: isize) {
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

    fn open_key(&mut self) {
        if self.tab != Tab::Autotune || self.detail.is_some() {
            return;
        }
        let key = self
            .pages
            .as_ref()
            .zip(self.keys.selected())
            .and_then(|(pages, index)| pages.autotune.keys.get(index));
        if let Some(key) = key {
            self.detail = Some(Text(key).to_string());
            self.scroll = 0;
        }
    }

    fn refresh_if_changed(&mut self) {
        self.checked = Instant::now();
        if self.seen != Some(Fingerprint::of(&self.path)) {
            self.refresh();
        }
    }

    /// Read every report again. A file mid-write that cannot be read keeps
    /// the previous reading on screen, and says why.
    fn refresh(&mut self) {
        self.seen = Some(Fingerprint::of(&self.path));
        match Inspector::open(&self.path) {
            Ok(inspector) => {
                self.pages = Some(Pages::read(&inspector));
                self.status = format!("read {}", clock(SystemTime::now()));
            }
            Err(err) => self.status = err.to_string(),
        }
    }

    fn draw(&mut self, frame: &mut Frame) {
        let [tabs, body, status] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Fill(1),
            Constraint::Length(1),
        ])
        .areas(frame.area());

        frame.render_widget(
            Tabs::new(Tab::ALL.map(Tab::title))
                .select(self.tab.index())
                .highlight_style(Style::new().add_modifier(Modifier::REVERSED)),
            tabs,
        );

        let block = Block::bordered().title(self.path.display().to_string());
        match (&self.pages, self.tab, &self.detail) {
            (None, _, _) => frame.render_widget(Paragraph::new("reading…").block(block), body),
            (Some(_), Tab::Autotune, Some(detail)) => frame.render_widget(
                Paragraph::new(detail.as_str())
                    .block(block)
                    .scroll((self.scroll, 0)),
                body,
            ),
            (Some(pages), Tab::Autotune, None) => {
                let items: Vec<ListItem> = pages
                    .autotune
                    .keys
                    .iter()
                    .map(|key| {
                        ListItem::new(format!(
                            "{}  {:>9}  {}  {}",
                            key.id,
                            Maybe(key.wall().map(Wall)).to_string(),
                            key.table.tuner,
                            key.winner_name()
                        ))
                    })
                    .collect();
                frame.render_stateful_widget(
                    List::new(items)
                        .block(block)
                        .highlight_style(Style::new().add_modifier(Modifier::REVERSED)),
                    body,
                    &mut self.keys,
                );
            }
            (Some(pages), tab, _) => frame.render_widget(
                Paragraph::new(pages.text(tab))
                    .block(block)
                    .scroll((self.scroll, 0)),
                body,
            ),
        }

        frame.render_widget(
            Line::from(format!(
                " {}  ·  ←→ tab  ↑↓ PgUp PgDn move  ⏎ open  esc back  r read  q quit",
                self.status
            )),
            status,
        );
    }
}

/// A wall-clock time of day, UTC, to the second.
fn clock(time: SystemTime) -> String {
    let secs = time
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |since| since.as_secs())
        % 86_400;
    format!(
        "{:02}:{:02}:{:02} UTC",
        secs / 3600,
        secs % 3600 / 60,
        secs % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::crossterm::event::KeyModifiers;

    fn press(code: KeyCode) -> Action {
        Action::from(KeyEvent::new(code, KeyModifiers::NONE))
    }

    #[test]
    fn tabs_wrap_both_ways() {
        assert_eq!(Tab::Summary.step(-1), Tab::Memory);
        assert_eq!(Tab::Memory.step(1), Tab::Summary);
        assert_eq!(Tab::Timeline.step(1), Tab::Autotune);
    }

    #[test]
    fn keys_name_their_actions() {
        assert_eq!(press(KeyCode::Char('q')), Action::Quit);
        assert_eq!(press(KeyCode::PageDown), Action::Down(PAGE));
        assert_eq!(press(KeyCode::Enter), Action::Open);
        assert_eq!(press(KeyCode::Char('x')), Action::Nothing);
    }

    /// The selection stays on the list, whatever the key asks.
    #[test]
    fn the_selection_is_clamped_to_the_keys() {
        let mut explorer = Explorer::new(PathBuf::from("missing.cubecl"));
        explorer.tab = Tab::Autotune;
        explorer.apply(Action::Up(5));
        assert_eq!(explorer.keys.selected(), Some(0));
        explorer.apply(Action::Down(5));
        assert_eq!(explorer.keys.selected(), Some(0), "no keys read");
    }
}
