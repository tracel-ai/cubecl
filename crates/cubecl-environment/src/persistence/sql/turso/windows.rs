//! Synchronous Windows file I/O with IOCP's shared-WAL coordination.
//!
//! Turso 0.8.2 publishes a completion group's zero outstanding count before
//! its result. With IOCP, another thread can reap the last write and expose
//! that gap to the committing thread, which reports a spurious I/O error.
//! `WindowsIO` completes data/WAL operations on the calling thread instead.
//! Only the coordination sidecar needs IOCP's byte locks and mappings; its
//! size, mapping, locking and sync operations complete synchronously too.

use alloc::string::String;
use alloc::sync::Arc;
use turso::core::io::{WindowsIO, WindowsIOCP};
use turso::core::{Clock, File, IO, MonotonicInstant, OpenFlags, Result, WallClockInstant};

pub(super) struct WindowsIo {
    files: WindowsIO,
    coordination: WindowsIOCP,
    coordination_path: String,
}

impl WindowsIo {
    pub(super) fn new(location: &str) -> Result<Self> {
        Ok(Self {
            files: WindowsIO::new()?,
            coordination: WindowsIOCP::new()?,
            coordination_path: alloc::format!("{location}-tshm"),
        })
    }
}

impl IO for WindowsIo {
    fn open_file(&self, path: &str, flags: OpenFlags, direct: bool) -> Result<Arc<dyn File>> {
        // Read-only opens reach an existing sidecar through open_file, not
        // open_shared_wal_file. Preserve their flags so a missing sidecar
        // stays missing. Match the exact path, since a database itself may
        // have a name ending in "-tshm".
        if path == self.coordination_path {
            self.coordination.open_file(path, flags, direct)
        } else {
            self.files.open_file(path, flags, direct)
        }
    }

    fn open_shared_wal_file(&self, path: &str) -> Result<Arc<dyn File>> {
        self.coordination.open_shared_wal_file(path)
    }

    fn supports_shared_wal_coordination(&self) -> bool {
        self.coordination.supports_shared_wal_coordination()
    }

    fn remove_file(&self, path: &str) -> Result<()> {
        self.files.remove_file(path)
    }
}

impl Clock for WindowsIo {
    fn current_time_monotonic(&self) -> MonotonicInstant {
        self.files.current_time_monotonic()
    }

    fn current_time_wall_clock(&self) -> WallClockInstant {
        self.files.current_time_wall_clock()
    }
}
