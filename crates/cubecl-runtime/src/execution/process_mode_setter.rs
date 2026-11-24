use super::{OPEN_MODE, ProcessMode, StatisticsCollector, StatisticsRecorder};
use alloc::format;
use alloc::string::String;
use cubecl_environment::sync::Mutex;

/// Opens and closes the process mode the live [`ProcessModeOverride`](super::ProcessModeOverride)s apply.
pub struct ProcessModeSetter {
    /// The open override, while one is. Every write of [`OPEN_MODE`] happens
    /// under its lock.
    open: Mutex<Option<OpenOverride>>,
}

/// The process's process mode.
pub static PROCESS_MODES: ProcessModeSetter = ProcessModeSetter {
    open: Mutex::new(None),
};

impl ProcessModeSetter {
    /// Applies `mode`, counting into `collector`, until it is
    /// [unset](Self::unset).
    ///
    /// # Panics
    ///
    /// If an override of another process mode, or of another collector, is
    /// open, which is left as it was.
    pub fn set(&self, mode: ProcessMode, collector: &StatisticsCollector) {
        let mut open = self.open.lock();
        let opened = OpenOverride::opening(open.as_ref(), mode, collector)
            .unwrap_or_else(|refusal| panic!("{}", refusal.message()));
        *open = Some(opened);
        OPEN_MODE.store(Some(mode));
    }

    /// Where the work the open override triggers is tallied, or `None` when
    /// none is open: what a [`Registration`](super::Registration) holds.
    ///
    /// Read without the lock first, then under it: a launch issued as one
    /// override closes while another collector's opens on another thread
    /// counts to the one opening.
    pub fn recorder(&self) -> Option<StatisticsRecorder> {
        OPEN_MODE.load()?;
        self.open.lock().as_ref().map(|open| open.recorder.clone())
    }

    /// Lets one override [set](Self::set) go: the last out closes the
    /// process mode, and its collector with it.
    pub fn unset(&self) {
        let mut open = self.open.lock();
        *open = open.as_ref().and_then(OpenOverride::left);
        OPEN_MODE.store(open.as_ref().map(|open| open.mode));
    }
}

/// The override open, as the [`ProcessModeSetter`] holds it.
#[derive(Debug, Clone)]
struct OpenOverride {
    /// Its process mode.
    pub mode: ProcessMode,
    /// Where the work under it is tallied.
    pub recorder: StatisticsRecorder,
    /// How many [`ProcessModeOverride`](super::ProcessModeOverride)s hold it open.
    pub guard_count: usize,
}

impl OpenOverride {
    /// The override open once one of `mode` and `collector` opens beside
    /// `open`, or why it cannot.
    fn opening(
        open: Option<&Self>,
        mode: ProcessMode,
        collector: &StatisticsCollector,
    ) -> Result<Self, OpenRefusal> {
        let Some(open) = open else {
            return Ok(Self {
                mode,
                recorder: collector.to_recorder(),
                guard_count: 1,
            });
        };
        match (open.recorder.counts_into(collector), open.mode == mode) {
            (false, _) => Err(OpenRefusal::OtherCollector),
            (true, false) => Err(OpenRefusal::OtherMode {
                asked: mode,
                open: open.mode,
            }),
            (true, true) => Ok(Self {
                guard_count: open.guard_count + 1,
                ..open.clone()
            }),
        }
    }

    /// The override open once one of its guards drops, `None` with the last.
    fn left(&self) -> Option<Self> {
        match self.guard_count {
            1 => None,
            guard_count => Some(Self {
                guard_count: guard_count - 1,
                ..self.clone()
            }),
        }
    }
}

/// Why an override cannot open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpenRefusal {
    /// One of another collector is open.
    OtherCollector,
    /// One of its collector, of another process mode, is open.
    OtherMode {
        /// The process mode asked for.
        asked: ProcessMode,
        /// The process mode open.
        open: ProcessMode,
    },
}

impl OpenRefusal {
    /// What the refused open panics with.
    fn message(self) -> String {
        match self {
            Self::OtherCollector => {
                String::from("an override cannot open while one of another collector is")
            }
            Self::OtherMode { asked, open } => {
                format!("an override of {asked:?} cannot open while one of {open:?} is")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open(mode: ProcessMode, collector: &StatisticsCollector) -> OpenOverride {
        OpenOverride {
            mode,
            recorder: collector.to_recorder(),
            guard_count: 1,
        }
    }

    #[test]
    fn an_override_opens_alone_with_one_guard() {
        let collector = StatisticsCollector::new();
        let opened = OpenOverride::opening(None, ProcessMode::CompileOnly, &collector);
        assert_eq!(opened.map(|opened| opened.guard_count), Ok(1));
    }

    #[test]
    fn an_override_of_the_same_mode_and_collector_joins() {
        let collector = StatisticsCollector::new();
        let held = open(ProcessMode::CompileOnly, &collector);
        let joined = OpenOverride::opening(Some(&held), ProcessMode::CompileOnly, &collector);
        assert_eq!(joined.map(|joined| joined.guard_count), Ok(2));
    }

    #[test]
    fn an_override_of_another_collector_is_refused() {
        let held = open(ProcessMode::CompileOnly, &StatisticsCollector::new());
        let refused = OpenOverride::opening(
            Some(&held),
            ProcessMode::CompileOnly,
            &StatisticsCollector::new(),
        );
        assert_eq!(refused.err(), Some(OpenRefusal::OtherCollector));
    }

    #[test]
    fn an_override_of_another_mode_is_refused() {
        let collector = StatisticsCollector::new();
        let held = open(ProcessMode::CompileAndAutotune, &collector);
        let refused = OpenOverride::opening(Some(&held), ProcessMode::CompileOnly, &collector);
        assert_eq!(
            refused.err(),
            Some(OpenRefusal::OtherMode {
                asked: ProcessMode::CompileOnly,
                open: ProcessMode::CompileAndAutotune,
            })
        );
    }

    #[test]
    fn the_last_guard_out_closes_the_override() {
        let held = open(ProcessMode::CompileOnly, &StatisticsCollector::new());
        assert!(held.left().is_none());
    }

    #[test]
    fn a_guard_out_before_the_last_leaves_the_override_open() {
        let held = OpenOverride {
            guard_count: 2,
            ..open(ProcessMode::CompileOnly, &StatisticsCollector::new())
        };
        assert_eq!(held.left().map(|left| left.guard_count), Some(1));
    }
}
