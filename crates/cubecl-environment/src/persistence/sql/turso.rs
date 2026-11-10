//! Turso under an environment's database.
//!
//! The engine's API is a set of futures, but its work is synchronous: a
//! statement steps, asks its I/O for a page, and steps again once the I/O has
//! answered. Natively the file system answers within the call, and the
//! browser's files are synchronous access handles that do the same, so a
//! statement never waits on anything an event loop would deliver. Every
//! statement here is driven to completion on the spot ([`drive`]), and the
//! calls this engine answers are synchronous like the other engine's.
//!
//! Opening is the exception: the browser reaches its files through promises,
//! so [`Database::open`] stays `async`.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::future::Future;
use core::time::Duration;

use super::{Opening, Param, SqlError};

#[cfg(all(windows, not(miri)))]
mod windows;

/// The engine, as a message names it.
pub(crate) const ENGINE: &str = "Turso";

/// The frames of a log turso indexes between the processes sharing its file,
/// past which every read of the file fails as `database is locked` until the
/// log restarts.
#[cfg_attr(
    browser_cache,
    expect(dead_code, reason = "one tab writes the browser's files")
)]
pub(crate) const INDEXED_LOG_FRAMES: Option<u64> = Some(262_144);

/// A database file opened by the engine, which every connection to the file
/// in this process shares.
#[derive(Debug)]
pub(crate) struct Database(::turso::Database);

impl Database {
    /// Opens the file at `location` as `opening` says.
    #[cfg(native_cache)]
    pub(crate) async fn open(location: &str, opening: Opening) -> Result<Self, SqlError> {
        // The SDK's process-local lock does not exclude IOCP shared-WAL
        // owners. Refuse this mode rather than promise recovery exclusivity.
        #[cfg(windows)]
        if opening == Opening::ProcessLocal {
            return Err(SqlError::Other(
                "process-local Turso recovery is unavailable on Windows; use a coordinated checkpoint"
                    .into(),
            ));
        }
        let opened = match opening {
            Opening::ReadOnly => open_read_only(location).await,
            Opening::ProcessLocal => {
                native_builder(location)?
                    .experimental_multiprocess_wal(false)
                    .build()
                    .await
            }
            Opening::Staging => {
                native_builder(location)?
                    .experimental_vacuum(true)
                    .build()
                    .await
            }
            Opening::Shared => open_shared(location).await,
        };
        opened.map(Self).map_err(SqlError::from)
    }

    /// Opens the browser's file at `location`, which this tab alone reads and
    /// writes, whatever `opening` says.
    #[cfg(browser_cache)]
    pub(crate) async fn open(location: &str, opening: Opening) -> Result<Self, SqlError> {
        let _ = opening;
        let wal = alloc::format!("{location}-wal");
        let io = crate::persistence::turso_browser::BrowserIo::new(&[location, &wal])
            .await
            .map_err(SqlError::Other)?;
        ::turso::Builder::new_local(location)
            .with_io_impl(alloc::sync::Arc::new(io))
            .build()
            .await
            .map(Self)
            .map_err(SqlError::from)
    }

    /// A new connection to the file.
    pub(crate) fn connect(&self) -> Result<Connection, SqlError> {
        self.0.connect().map(Connection).map_err(SqlError::from)
    }
}

/// A builder for the native file at `location`.
///
/// Windows keeps synchronous data I/O while using IOCP's shared-WAL locks
/// and mappings. In-memory databases retain the default memory backend.
#[cfg(native_cache)]
fn native_builder(location: &str) -> Result<::turso::Builder, ::turso::Error> {
    let builder = ::turso::Builder::new_local(location);
    #[cfg(all(windows, not(miri)))]
    let builder = if location == ":memory:" {
        builder
    } else {
        let io = windows::WindowsIo::new(location)
            .map_err(|error| ::turso::Error::Error(error.to_string()))?;
        builder.with_io_impl(alloc::sync::Arc::new(io))
    };
    Ok(builder)
}

/// Join a live writer's WAL, without creating a missing coordination file.
/// Only locations that cannot support shared WAL may use the legacy reader;
/// retrying other failures without coordination could serve stale data.
#[cfg(native_cache)]
async fn open_read_only(location: &str) -> Result<::turso::Database, ::turso::Error> {
    let opened = native_builder(location)?
        .experimental_multiprocess_wal(true)
        .read_only(true)
        .build()
        .await;
    match opened {
        Err(err) if multiprocess_wal_unsupported(&err) => {
            native_builder(location)?.read_only(true).build().await
        }
        opened => opened,
    }
}

/// Opens `location` with its WAL shared between processes, or for this
/// process alone where the engine won't share it.
///
/// Writable for this process alone is still a cache that outlives the run.
/// Passed up, the refusal would land in the read-only fallback instead: a new
/// file doesn't open there at all, and an existing one opens unable to take a
/// single write.
#[cfg(native_cache)]
async fn open_shared(location: &str) -> Result<::turso::Database, ::turso::Error> {
    match open_writable(location, true).await {
        Err(err) if multiprocess_wal_unsupported(&err) => {
            log::info!("cubecl cache: {err}; opening {location} for this process alone");
            open_writable(location, false).await
        }
        opened => opened,
    }
}

/// Opens `location` writable, its WAL shared between processes when
/// `multiprocess`.
///
/// Without it, the engine uses its legacy process lock. This fallback is for
/// locations that cannot support shared WAL, not for recovery beside shared
/// owners: the legacy lock does not exclude those owners on Windows.
#[cfg(native_cache)]
async fn open_writable(
    location: &str,
    multiprocess: bool,
) -> Result<::turso::Database, ::turso::Error> {
    native_builder(location)?
        .experimental_multiprocess_wal(multiprocess)
        .build()
        .await
}

/// Whether `err` is the engine declining to share a file's WAL between
/// processes.
///
/// It declines wherever it can't coordinate the WAL through shared memory: on
/// network and cluster file systems (NFS, SMB, Lustre, Ceph, 9p, ...), whose
/// shared memory isn't coherent between hosts, which on Windows is anything
/// but a fixed, removable or RAM drive; and for in-memory databases. Every one
/// of them reaches the SDK as a plain [`::turso::Error::Error`], the variant
/// most engine errors map to, so the message is the only thing that sets them
/// apart from a failure that single-process mode would hit just the same.
#[cfg(native_cache)]
fn multiprocess_wal_unsupported(err: &::turso::Error) -> bool {
    matches!(err, ::turso::Error::Error(message) if message.contains("multiprocess WAL is not supported"))
}

/// Whether `message` is the engine refusing to open a file for this process
/// alone because another process has it open.
///
/// The engine reports it as a locking error, which reaches the SDK as a plain
/// [`::turso::Error::Error`] like most of its errors: the wording is what sets
/// it apart, as for [`multiprocess_wal_unsupported`].
fn held_by_another_process(message: &str) -> bool {
    message.contains("in another process") || message.contains("locked by another process")
}

impl From<::turso::Error> for SqlError {
    fn from(error: ::turso::Error) -> Self {
        let message = error.to_string();
        match error {
            ::turso::Error::Busy(_) | ::turso::Error::BusySnapshot(_) => Self::Busy(message),
            ::turso::Error::Error(_) if held_by_another_process(&message) => {
                Self::HeldByAnotherProcess(message)
            }
            ::turso::Error::Corrupt(_) => Self::Corrupt(message),
            ::turso::Error::NotAdb(_) => Self::NotADatabase(message),
            ::turso::Error::Readonly(_) => Self::ReadOnly(message),
            ::turso::Error::DatabaseFull(_) => Self::Full(message),
            ::turso::Error::IoError(..) => Self::Io(message),
            _ => Self::Other(message),
        }
    }
}

/// A connection to a database file.
pub(crate) struct Connection(::turso::Connection);

impl Connection {
    /// Runs `sql`, returning how many rows it changed.
    pub(crate) fn execute(&self, sql: &str, params: &[Param<'_>]) -> Result<u64, SqlError> {
        drive(self.0.execute(sql, values(params))).map_err(SqlError::from)
    }

    /// Runs `sql`, visiting each row it returns, in order.
    pub(crate) fn query<E: From<SqlError>>(
        &self,
        sql: &str,
        params: &[Param<'_>],
        visit: &mut dyn FnMut(&Row<'_>) -> Result<(), E>,
    ) -> Result<(), E> {
        let mut rows = drive(self.0.query(sql, values(params))).map_err(SqlError::from)?;
        while let Some(row) = drive(rows.next()).map_err(SqlError::from)? {
            visit(&Row(&row))?;
        }
        Ok(())
    }

    /// Runs `PRAGMA <pragma>`, visiting each row it returns. An assignment
    /// answers with no rows, which [`execute`](Self::execute) would refuse.
    pub(crate) fn pragma(
        &self,
        pragma: &str,
        visit: &mut dyn FnMut(&Row<'_>),
    ) -> Result<(), SqlError> {
        drive(self.0.pragma_query(pragma, |row| {
            visit(&Row(row));
            Ok(())
        }))
        .map_err(SqlError::from)
    }

    /// How long a statement waits on a lock another connection holds before
    /// it fails as [`SqlError::Busy`].
    pub(crate) fn busy_timeout(&self, timeout: Duration) -> Result<(), SqlError> {
        self.0.busy_timeout(timeout).map_err(SqlError::from)
    }

    /// Folds the log into the file and restarts it from its first frame.
    ///
    /// Reports whether the checkpoint completed; it doesn't when another
    /// connection holds the log, and Turso folds every other cause into the
    /// same flag while logging the reason itself.
    #[cfg(native_cache)]
    pub(crate) fn checkpoint(&self) -> Result<bool, SqlError> {
        let mut complete = true;
        self.pragma("wal_checkpoint(TRUNCATE)", &mut |row| {
            // A row that doesn't decode is read as "did not complete".
            complete &= row.integer(0).is_ok_and(|busy| busy == 0);
        })?;
        Ok(complete)
    }
}

/// One row a query returned.
pub(crate) struct Row<'r>(&'r ::turso::Row);

impl Row<'_> {
    /// The text of column `index`.
    pub(crate) fn text(&self, index: usize) -> Result<String, SqlError> {
        self.0.get(index).map_err(SqlError::from)
    }

    /// The bytes of column `index`.
    pub(crate) fn blob(&self, index: usize) -> Result<Vec<u8>, SqlError> {
        self.0.get(index).map_err(SqlError::from)
    }

    /// The integer of column `index`.
    pub(crate) fn integer(&self, index: usize) -> Result<i64, SqlError> {
        self.0.get(index).map_err(SqlError::from)
    }
}

/// `params` as the engine binds them.
fn values(params: &[Param<'_>]) -> Vec<::turso::Value> {
    params
        .iter()
        .map(|param| match *param {
            Param::Text(text) => ::turso::Value::Text(text.to_string()),
            Param::Blob(blob) => ::turso::Value::Blob(blob.to_vec()),
            Param::Integer(integer) => ::turso::Value::Integer(integer),
        })
        .collect()
}

/// Runs one of the engine's futures to completion.
///
/// Natively this blocks the thread, as reading a file would. In the browser
/// nothing can block, and nothing has to: an engine future is pending only
/// between a step that asked for I/O and the next, and the browser's files
/// ([`crate::persistence::turso_browser`]) answer within the call, so the
/// next poll finds the answer. Polling in a loop is therefore how it is
/// driven — and why a future that awaits the event loop must never come
/// through here. Opening the database does, and stays `async`.
fn drive<T>(future: impl Future<Output = T>) -> T {
    #[cfg(not(browser_cache))]
    {
        crate::future::block_on(future)
    }

    #[cfg(browser_cache)]
    {
        use core::pin::pin;
        use core::task::{Context, Poll, Waker};

        let mut future = pin!(future);
        let mut context = Context::from_waker(Waker::noop());
        loop {
            if let Poll::Ready(output) = future.as_mut().poll(&mut context) {
                return output;
            }
        }
    }
}

#[cfg(all(test, native_cache))]
mod tests {
    use super::*;
    use crate::future::block_on;

    /// Where the engine won't share a file's WAL between processes, the file
    /// still opens writable, for this process alone, rather than failing into
    /// the read-only fallback, which can't open a file that isn't an
    /// environment yet.
    ///
    /// Network file systems are where this matters, and CI runs on none. An
    /// in-memory location is declined by the same check, with the same error,
    /// on every platform.
    #[test_log::test]
    #[cfg_attr(miri, ignore)]
    fn a_location_without_a_shared_wal_opens_for_this_process() {
        let database = block_on(Database::open(":memory:", Opening::Shared)).unwrap();
        let connection = database.connect().unwrap();
        connection
            .execute("CREATE TABLE notes (body TEXT)", &[])
            .unwrap();
        assert_eq!(
            connection
                .execute("INSERT INTO notes VALUES (?1)", &["written".into()])
                .unwrap(),
            1
        );
    }

    /// A file on a local disk is shared between processes: it gets the index
    /// file (`-tshm`) the engine coordinates the WAL through. On Windows this
    /// takes the IOCP backend; the default `WindowsIO` refuses a shared WAL, and
    /// the file used to open for one process only.
    #[test_log::test]
    #[cfg_attr(miri, ignore)]
    #[cfg(all(any(unix, windows), target_pointer_width = "64"))]
    fn a_local_file_opens_with_a_shared_wal() {
        let dir = tempfile::tempdir().unwrap();
        let location = dir.path().join("cache.db");
        let location = location.to_str().unwrap();

        let database = block_on(Database::open(location, Opening::Shared)).unwrap();
        let connection = database.connect().unwrap();
        connection
            .execute("CREATE TABLE notes (body TEXT)", &[])
            .unwrap();
        connection
            .execute("INSERT INTO notes VALUES (?1)", &["written".into()])
            .unwrap();

        assert!(
            std::path::Path::new(&alloc::format!("{location}-tshm")).exists(),
            "{location} opened without a shared WAL"
        );
    }

    /// The fallback keys on the engine's message, so it is pinned here: an
    /// engine that rewords the refusal must fail this test, not quietly send
    /// every such location back to the read-only fallback.
    ///
    /// Only where the engine coordinates WALs at all, 64-bit Unix and Windows;
    /// elsewhere it ignores the request rather than refusing it.
    #[test_log::test]
    #[cfg_attr(miri, ignore)]
    #[cfg(all(any(unix, windows), target_pointer_width = "64"))]
    fn the_engine_declines_a_shared_wal_in_words_we_recognize() {
        let Err(err) = block_on(open_writable(":memory:", true)) else {
            panic!("an in-memory database must not get a shared WAL");
        };
        assert!(
            multiprocess_wal_unsupported(&err),
            "unrecognized refusal: {err}"
        );
    }
}
