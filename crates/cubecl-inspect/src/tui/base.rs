use super::picker::Picker;
use super::viewer::Viewer;
use crate::report::Listing;
use crate::{EnvironmentDirectory, InspectError, Inspector};
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
use ratatui::{DefaultTerminal, Frame};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

/// How long the loop waits for a key before looking at the file again.
const POLL: Duration = Duration::from_millis(250);

/// How often the file is looked at for a change.
const REFRESH: Duration = Duration::from_secs(1);

/// Rows a page key moves.
const PAGE: u16 = 20;

/// What the browser offers: the directories whose environments it lists to
/// choose from, and the one it opens first.
pub struct Browsing {
    /// Where to look for environments.
    pub directories: Vec<EnvironmentDirectory>,
    /// The environment opened on start; `None` starts on the list.
    pub open: Option<PathBuf>,
}

/// Browse until the user quits.
pub fn run(browsing: Browsing) -> Result<(), InspectError> {
    let mut explorer = Explorer::new(browsing);
    let mut terminal = ratatui::init();
    let result = explorer.run(&mut terminal);
    ratatui::restore();
    result
}

/// The browser: the list of environments, and the one open over it.
///
/// The list is kept while an environment is open, so backing out of one
/// returns to it without reading every file again; `r` on the list reads
/// them again.
struct Explorer {
    directories: Vec<EnvironmentDirectory>,
    picker: Picker,
    /// The environment open; the list shows while there is none.
    viewer: Option<Viewer>,
    /// What happened last: a reading, or why one failed.
    status: String,
    checked: Instant,
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

impl Explorer {
    /// List the environments, then open the one asked for, if any.
    fn new(browsing: Browsing) -> Self {
        let Browsing { directories, open } = browsing;
        let mut explorer = Self {
            picker: Picker::new(&Listing::default()),
            directories,
            viewer: None,
            status: String::new(),
            checked: Instant::now(),
        };
        explorer.list();
        if let Some(path) = open {
            explorer.open(path);
        }
        explorer
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
                self.checked = Instant::now();
                if self.viewer.as_ref().is_some_and(Viewer::changed) {
                    self.read();
                }
            }
        }
    }

    fn apply(&mut self, action: Action) {
        let Some(viewer) = &mut self.viewer else {
            return self.choose(action);
        };
        match action {
            Action::NextTab => viewer.step_tab(1),
            Action::PreviousTab => viewer.step_tab(-1),
            Action::Up(rows) => viewer.move_by(-(rows as isize)),
            Action::Down(rows) => viewer.move_by(rows as isize),
            Action::Open => viewer.open_key(),
            Action::Close => self.close(),
            Action::Refresh => self.read(),
            Action::Quit | Action::Nothing => {}
        }
    }

    /// What a key does on the list of environments.
    fn choose(&mut self, action: Action) {
        match action {
            Action::Up(rows) => self.picker.move_by(-(rows as isize)),
            Action::Down(rows) => self.picker.move_by(rows as isize),
            Action::Open => {
                if let Some(path) = self.picker.selected() {
                    self.open(path.to_path_buf());
                }
            }
            Action::Refresh => self.list(),
            Action::NextTab
            | Action::PreviousTab
            | Action::Close
            | Action::Quit
            | Action::Nothing => {}
        }
    }

    /// Read the directories into the list again.
    fn list(&mut self) {
        self.picker = Picker::new(&Inspector::list(&self.directories));
        self.status = self.picker.status();
    }

    fn open(&mut self, path: PathBuf) {
        self.viewer = Some(Viewer::new(path));
        self.read();
    }

    /// Back out of the open key, or else out of the environment.
    fn close(&mut self) {
        if let Some(viewer) = &mut self.viewer
            && !viewer.close_key()
        {
            self.viewer = None;
            self.status = self.picker.status();
        }
    }

    /// Read the open environment again, saying how that went.
    fn read(&mut self) {
        let Some(viewer) = &mut self.viewer else {
            return;
        };
        self.status = match viewer.read() {
            Ok(()) => format!("read {}", clock(SystemTime::now())),
            Err(err) => err.to_string(),
        };
    }

    fn draw(&mut self, frame: &mut Frame) {
        match &mut self.viewer {
            Some(viewer) => viewer.draw(frame, &self.status),
            None => self.picker.draw(frame, &self.directories, &self.status),
        }
    }
}

/// A path the way a person reads it: under the home directory, from `~`.
pub(crate) fn shown(path: &Path) -> String {
    match std::env::home_dir().and_then(|home| path.strip_prefix(home).ok().map(Path::to_path_buf))
    {
        Some(relative) => format!("~/{}", relative.display()),
        None => path.display().to_string(),
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

    fn explorer(open: Option<&str>) -> Explorer {
        Explorer::new(Browsing {
            directories: Vec::new(),
            open: open.map(PathBuf::from),
        })
    }

    fn open(explorer: &Explorer) -> Option<&Path> {
        explorer.viewer.as_ref().map(Viewer::path)
    }

    #[test]
    fn keys_name_their_actions() {
        assert_eq!(press(KeyCode::Char('q')), Action::Quit);
        assert_eq!(press(KeyCode::PageDown), Action::Down(PAGE));
        assert_eq!(press(KeyCode::Enter), Action::Open);
        assert_eq!(press(KeyCode::Char('x')), Action::Nothing);
    }

    /// Backing out of an environment returns to the list, and opening from
    /// the list opens the one chosen.
    #[test]
    fn esc_returns_to_the_environments_and_enter_opens_one() {
        let mut explorer = explorer(Some("missing.cubecl"));
        assert_eq!(open(&explorer), Some(Path::new("missing.cubecl")));

        explorer.apply(Action::Close);
        assert_eq!(open(&explorer), None);

        let summary = |path: &str| crate::report::Summary {
            path: PathBuf::from(path),
            file_bytes: 0,
            manifest: None,
            namespaces: Vec::new(),
            sessions: Vec::new(),
        };
        explorer.picker = Picker::new(&Listing {
            environments: vec![summary("a.db"), summary("b.db")],
            ..Listing::default()
        });
        explorer.apply(Action::Down(5));
        explorer.apply(Action::Open);
        assert_eq!(open(&explorer), Some(Path::new("b.db")));
    }

    #[test]
    fn with_nothing_to_open_it_starts_on_the_list() {
        let mut explorer = explorer(None);
        assert!(explorer.viewer.is_none());
        explorer.apply(Action::Open);
        assert!(explorer.viewer.is_none(), "nothing listed, nothing opened");
    }
}
