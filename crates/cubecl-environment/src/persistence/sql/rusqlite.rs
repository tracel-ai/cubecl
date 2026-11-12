//! `SQLite`, through rusqlite, under an environment's native files: the
//! `rusqlite` feature.
//!
//! `SQLite` coordinates every process sharing a file through the file's own
//! locks and its `-shm` index, which has no cap on the frames a log may hold:
//! nothing here restarts a log to keep it readable, and a file opened for
//! this process alone is opened like any other.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::time::Duration;

use ::rusqlite::types::{ToSqlOutput, ValueRef};
use ::rusqlite::{ErrorCode, OpenFlags, ToSql};

use super::{Opening, Param, SqlError};

/// The engine, as a message names it.
pub(crate) const ENGINE: &str = "SQLite";

/// `SQLite` indexes a log of any length between processes.
pub(crate) const INDEXED_LOG_FRAMES: Option<u64> = None;

/// A database file, which each connection opens on its own: what to open,
/// and how.
#[derive(Debug)]
pub(crate) struct Database {
    path: String,
    flags: OpenFlags,
}

impl Database {
    /// Opens the file at `location` as `opening` says: a writable one is
    /// created when missing and put in WAL mode, which the file keeps.
    pub(crate) async fn open(location: &str, opening: Opening) -> Result<Self, SqlError> {
        match opening {
            Opening::ReadOnly => Self::open_read_only(location),
            Opening::Shared | Opening::ProcessLocal | Opening::Staging => {
                let database = Self {
                    path: location.to_string(),
                    flags: OpenFlags::default(),
                };
                database.open_connection()?.pragma_update_and_check(
                    None,
                    "journal_mode",
                    "WAL",
                    |_| Ok(()),
                )?;
                Ok(database)
            }
        }
    }

    /// Opens a file for reading, tolerating a directory this process may not
    /// write.
    ///
    /// A reader of a WAL file has to create the `-shm` index next to it, so a
    /// file in a container image layer, a Nix store or `/usr/share` fails to
    /// open at all. The URI `immutable=1` form skips locking and the index
    /// entirely, which is right for a file nobody can write, and wrong for one
    /// another process is appending to: it is only ever a fallback.
    fn open_read_only(location: &str) -> Result<Self, SqlError> {
        let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX;
        let in_place = Self {
            path: location.to_string(),
            flags,
        };
        // Opening is lazy; reading the schema is what creates the index, so
        // an open alone proves nothing.
        let read = in_place.open_connection().and_then(|connection| {
            connection.query_row("SELECT count(*) FROM sqlite_schema", [], |_| Ok(()))?;
            Ok(())
        });
        let Err(err) = read else {
            return Ok(in_place);
        };
        log::debug!(
            "cubecl cache: {location} is not readable in place ({err}); reading it immutable"
        );
        Ok(Self {
            path: immutable_uri(location),
            flags: flags | OpenFlags::SQLITE_OPEN_URI,
        })
    }

    /// A new connection to the file.
    pub(crate) fn connect(&self) -> Result<Connection, SqlError> {
        self.open_connection().map(Connection)
    }

    fn open_connection(&self) -> Result<::rusqlite::Connection, SqlError> {
        Ok(::rusqlite::Connection::open_with_flags(
            &self.path, self.flags,
        )?)
    }
}

/// The `file:` URI naming `location` as an immutable database.
fn immutable_uri(location: &str) -> String {
    let mut uri = String::from("file:");
    for character in location.chars() {
        match character {
            // The only characters that would end the path of a URI.
            '?' => uri.push_str("%3f"),
            '#' => uri.push_str("%23"),
            '%' => uri.push_str("%25"),
            _ => uri.push(character),
        }
    }
    uri.push_str("?immutable=1");
    uri
}

impl From<::rusqlite::Error> for SqlError {
    fn from(error: ::rusqlite::Error) -> Self {
        let message = error.to_string();
        let code = match &error {
            ::rusqlite::Error::SqliteFailure(failure, _) => Some(failure.code),
            _ => None,
        };
        match code {
            Some(ErrorCode::DatabaseBusy | ErrorCode::DatabaseLocked) => Self::Busy(message),
            Some(ErrorCode::DatabaseCorrupt) => Self::Corrupt(message),
            Some(ErrorCode::NotADatabase) => Self::NotADatabase(message),
            Some(ErrorCode::ReadOnly) => Self::ReadOnly(message),
            Some(ErrorCode::DiskFull) => Self::Full(message),
            Some(ErrorCode::SystemIoFailure | ErrorCode::CannotOpen) => Self::Io(message),
            _ => Self::Other(message),
        }
    }
}

impl ToSql for Param<'_> {
    fn to_sql(&self) -> ::rusqlite::Result<ToSqlOutput<'_>> {
        Ok(ToSqlOutput::Borrowed(match *self {
            Param::Text(text) => ValueRef::Text(text.as_bytes()),
            Param::Blob(blob) => ValueRef::Blob(blob),
            Param::Integer(integer) => ValueRef::Integer(integer),
        }))
    }
}

/// A connection to a database file.
pub(crate) struct Connection(::rusqlite::Connection);

impl Connection {
    /// Runs `sql`, returning how many rows it changed.
    pub(crate) fn execute(&self, sql: &str, params: &[Param<'_>]) -> Result<u64, SqlError> {
        let mut statement = self.0.prepare_cached(sql)?;
        let changed = statement.execute(::rusqlite::params_from_iter(params))?;
        Ok(changed as u64)
    }

    /// Runs `sql`, visiting each row it returns, in order.
    pub(crate) fn query<E: From<SqlError>>(
        &self,
        sql: &str,
        params: &[Param<'_>],
        visit: &mut dyn FnMut(&Row<'_>) -> Result<(), E>,
    ) -> Result<(), E> {
        let mut statement = self.0.prepare_cached(sql).map_err(SqlError::from)?;
        let mut rows = statement
            .query(::rusqlite::params_from_iter(params))
            .map_err(SqlError::from)?;
        while let Some(row) = rows.next().map_err(SqlError::from)? {
            visit(&Row(row))?;
        }
        Ok(())
    }

    /// Runs `PRAGMA <pragma>`, visiting each row it returns.
    pub(crate) fn pragma(
        &self,
        pragma: &str,
        visit: &mut dyn FnMut(&Row<'_>),
    ) -> Result<(), SqlError> {
        let mut statement = self.0.prepare(&alloc::format!("PRAGMA {pragma}"))?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            visit(&Row(row));
        }
        Ok(())
    }

    /// How long a statement waits on a lock another connection holds before
    /// it fails as [`SqlError::Busy`].
    pub(crate) fn busy_timeout(&self, timeout: Duration) -> Result<(), SqlError> {
        Ok(self.0.busy_timeout(timeout)?)
    }

    /// Folds the log into the file and truncates it.
    ///
    /// Reports whether the checkpoint completed; it doesn't when another
    /// connection holds the log.
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
pub(crate) struct Row<'r>(&'r ::rusqlite::Row<'r>);

impl Row<'_> {
    /// The text of column `index`.
    pub(crate) fn text(&self, index: usize) -> Result<String, SqlError> {
        Ok(self.0.get(index)?)
    }

    /// The bytes of column `index`.
    pub(crate) fn blob(&self, index: usize) -> Result<Vec<u8>, SqlError> {
        Ok(self.0.get(index)?)
    }

    /// The integer of column `index`.
    pub(crate) fn integer(&self, index: usize) -> Result<i64, SqlError> {
        Ok(self.0.get(index)?)
    }
}
