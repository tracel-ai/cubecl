use super::ProcessMode;
use cubecl_environment::sync::{AtomicUsize, Ordering};

/// The open process mode, as a launch reads it without a lock.
pub struct OpenMode {
    /// The open mode's [level](level_of), or
    /// [`CLOSED_LEVEL`] when none is open.
    level: AtomicUsize,
}

/// The level of an [`OpenMode`] with no process mode open.
const CLOSED_LEVEL: usize = 0;

/// [`OpenMode`] as the process holds it, written only under the lock of
/// [`PROCESS_MODES`](super::PROCESS_MODES).
pub static OPEN_MODE: OpenMode = OpenMode {
    level: AtomicUsize::new(CLOSED_LEVEL),
};

impl OpenMode {
    /// The open process mode, `None` when none is.
    pub fn load(&self) -> Option<ProcessMode> {
        mode_at_level(self.level.load(Ordering::Relaxed))
    }

    /// Opens `mode`, or with `None` closes the open one.
    pub fn store(&self, mode: Option<ProcessMode>) {
        self.level
            .store(mode.map_or(CLOSED_LEVEL, level_of), Ordering::Relaxed);
    }
}

/// The process mode an [`OpenMode`] holding `level` holds, `None` for none
/// open.
fn mode_at_level(level: usize) -> Option<ProcessMode> {
    match level {
        1 => Some(ProcessMode::CompileOnly),
        2 => Some(ProcessMode::CompileAndAutotune),
        3 => Some(ProcessMode::Execute),
        _ => None,
    }
}

/// The level an [`OpenMode`] holds `mode` as. Every process mode has a level
/// of its own, [`Execute`](ProcessMode::Execute) included, so an override of
/// it is told apart from none: it tallies what a real run compiles and
/// tunes.
fn level_of(mode: ProcessMode) -> usize {
    match mode {
        ProcessMode::CompileOnly => 1,
        ProcessMode::CompileAndAutotune => 2,
        ProcessMode::Execute => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_process_mode_has_a_level_of_its_own() {
        let modes = [
            ProcessMode::Execute,
            ProcessMode::CompileOnly,
            ProcessMode::CompileAndAutotune,
        ];
        assert_eq!(
            modes.map(|mode| mode_at_level(level_of(mode))),
            modes.map(Some)
        );
    }

    #[test]
    fn the_closed_level_holds_no_process_mode() {
        assert_eq!(mode_at_level(CLOSED_LEVEL), None);
    }

    #[test]
    fn an_open_mode_reads_the_mode_it_stored() {
        let open = OpenMode {
            level: AtomicUsize::new(CLOSED_LEVEL),
        };
        open.store(Some(ProcessMode::CompileOnly));
        assert_eq!(open.load(), Some(ProcessMode::CompileOnly));
    }
}
