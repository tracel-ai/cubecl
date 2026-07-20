//! `SQLite`-backed persistence.
//!
//! One database file per cache root holds every namespace, told apart by a
//! column instead of by a directory tree. That makes lookups per key rather
//! than per file, gives multi-process safety through WAL instead of a lock
//! sidecar, and turns bundle export into a query (see [`crate::bundle`]).

use std::path::{Path, PathBuf};
use std::string::{String, ToString};
use std::vec::Vec;

use hashbrown::HashMap;
use rusqlite::{Connection, OpenFlags, OptionalExtension, TransactionBehavior, params};

use super::storage::Storage;
use crate::sync::{Arc, Lazy, Mutex};

/// File name of the cache database inside a cache root.
pub const DB_FILE_NAME: &str = "cubecl.db";

/// The database schema this build reads and writes. A file carrying any other
/// version has its entries table dropped and rebuilt: it is a cache, so the
/// only cost is one cold start. Bump this on any change to
/// [`CREATE_ENTRIES`], including a renamed column.
pub const SCHEMA_VERSION: u32 = 2;

/// Created first and never dropped, so the schema version survives a rebuild
/// of the entries table.
const CREATE_META: &str = "
CREATE TABLE IF NOT EXISTS meta (
    k TEXT PRIMARY KEY,
    v TEXT NOT NULL
);
";

const CREATE_ENTRIES: &str = "
CREATE TABLE IF NOT EXISTS entries (
    namespace TEXT NOT NULL,
    key   BLOB NOT NULL,
    value BLOB NOT NULL,
    PRIMARY KEY (namespace, key)
);
";

const INSERT_SQL: &str =
    "INSERT INTO entries (namespace, key, value) VALUES (?1, ?2, ?3) ON CONFLICT DO NOTHING";
const SELECT_SQL: &str = "SELECT value FROM entries WHERE namespace = ?1 AND key = ?2";
const SCAN_SQL: &str = "SELECT key, value FROM entries WHERE namespace = ?1";

/// One open database file, shared by every namespace that lives in it.
#[derive(Clone)]
pub struct Database {
    conn: Arc<Mutex<Connection>>,
    path: Arc<PathBuf>,
}

impl core::fmt::Debug for Database {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Database({:?})", self.path)
    }
}

/// Databases already opened by this process, so N namespaces over one cache
/// root share a single connection.
static OPENED: Lazy<Mutex<HashMap<PathBuf, Database>>> = Lazy::new(|| Mutex::new(HashMap::new()));

impl Database {
    /// Opens (creating it if needed) the cache database of a cache root,
    /// reusing the connection when this process already opened it.
    ///
    /// Returns `None` when the database can't be opened — a read-only mount, a
    /// sandboxed application bundle, a missing parent directory. Callers fall
    /// back to memory-only persistence rather than failing.
    pub fn open_root(root: &Path) -> Option<Self> {
        let path = root.join(DB_FILE_NAME);

        let mut opened = OPENED.lock().expect("Lock recovers from poisoning");
        if let Some(database) = opened.get(&path) {
            return Some(database.clone());
        }

        if let Err(err) = std::fs::create_dir_all(root) {
            log::error!(
                "cubecl cache: create_dir_all({root:?}) failed: {err}; \
                 persistence is disabled for this root"
            );
            return None;
        }

        match Self::open(&path, false) {
            Ok(database) => {
                opened.insert(path, database.clone());
                Some(database)
            }
            Err(err) => {
                log::error!(
                    "cubecl cache: can't open {path:?}: {err}; \
                     persistence is disabled for this root"
                );
                None
            }
        }
    }

    /// Opens a database file directly, without consulting the process-wide
    /// registry. A read-only database is never written to nor migrated.
    pub fn open(path: &Path, read_only: bool) -> Result<Self, rusqlite::Error> {
        let flags = if read_only {
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX
        } else {
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_CREATE
                | OpenFlags::SQLITE_OPEN_NO_MUTEX
        };
        let conn = Connection::open_with_flags(path, flags)?;

        // WAL lets readers and one writer proceed concurrently across
        // processes, which is what the previous lock sidecar approximated.
        // `NORMAL` may lose the last commits on a power cut, which for a cache
        // costs a recompute.
        conn.busy_timeout(core::time::Duration::from_secs(5))?;
        if !read_only {
            conn.pragma_update(None, "journal_mode", "WAL")?;
            conn.pragma_update(None, "synchronous", "NORMAL")?;
            // Order matters: `meta` carries the schema version, `migrate` may
            // drop a stale `entries` table, and only then is it safe to create
            // one with the current column layout.
            conn.execute_batch(CREATE_META)?;
            migrate(&conn)?;
            conn.execute_batch(CREATE_ENTRIES)?;
        }

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            path: Arc::new(path.to_path_buf()),
        })
    }

    /// The database file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Runs `func` with the connection locked.
    pub fn with_connection<T>(&self, func: impl FnOnce(&mut Connection) -> T) -> T {
        let mut guard = self.conn.lock().expect("Lock recovers from poisoning");
        func(&mut guard)
    }

    /// The value stored under `key` in `namespace`.
    pub fn get(&self, namespace: &str, key: &[u8]) -> Option<Vec<u8>> {
        self.with_connection(|conn| {
            conn.prepare_cached(SELECT_SQL)
                .and_then(|mut stmt| {
                    stmt.query_row(params![namespace, key], |row| row.get(0))
                        .optional()
                })
                .unwrap_or_else(|err| {
                    self.warn("read", err);
                    None
                })
        })
    }

    /// Stores `value` unless `key` is already present in `namespace`, in which
    /// case the existing value is returned untouched.
    pub fn insert(&self, namespace: &str, key: &[u8], value: &[u8]) -> Option<Vec<u8>> {
        let result = self.with_connection(|conn| {
            let transaction = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
            let written = transaction
                .prepare_cached(INSERT_SQL)?
                .execute(params![namespace, key, value])?;

            let existing = if written == 0 {
                transaction
                    .prepare_cached(SELECT_SQL)?
                    .query_row(params![namespace, key], |row| row.get(0))
                    .optional()?
            } else {
                None
            };

            transaction.commit()?;
            Ok::<_, rusqlite::Error>(existing)
        });

        match result {
            Ok(existing) => existing,
            Err(err) => {
                // A dropped write degrades to a recompute on the next run.
                self.warn("write", err);
                None
            }
        }
    }

    /// Visits every entry of `namespace`.
    pub fn scan(&self, namespace: &str, visit: &mut dyn FnMut(Vec<u8>, Vec<u8>)) {
        let result = self.with_connection(|conn| {
            let mut stmt = conn.prepare_cached(SCAN_SQL)?;
            let mut rows = stmt.query(params![namespace])?;
            while let Some(row) = rows.next()? {
                visit(row.get(0)?, row.get(1)?);
            }
            Ok::<_, rusqlite::Error>(())
        });

        if let Err(err) = result {
            self.warn("scan", err);
        }
    }

    /// Entry count and total value size per namespace, for reporting.
    pub fn summary(&self) -> Vec<NamespaceSummary> {
        let result = self.with_connection(|conn| {
            let mut stmt = conn.prepare(
                "SELECT namespace, count(*), sum(length(key) + length(value)) \
                 FROM entries GROUP BY namespace ORDER BY namespace",
            )?;
            let rows = stmt.query_map([], |row| {
                Ok(NamespaceSummary {
                    namespace: row.get(0)?,
                    entries: row.get::<_, i64>(1)? as u64,
                    bytes: row.get::<_, Option<i64>>(2)?.unwrap_or(0) as u64,
                })
            })?;
            rows.collect::<Result<Vec<_>, _>>()
        });

        result.unwrap_or_else(|err| {
            self.warn("summary", err);
            Vec::new()
        })
    }

    fn warn(&self, operation: &str, err: rusqlite::Error) {
        log::warn!("cubecl cache: {operation} on {:?} failed: {err}", self.path);
    }
}

/// One namespace's contribution to a database, as reported by
/// [`Database::summary`].
#[derive(Debug, Clone)]
pub struct NamespaceSummary {
    /// The namespace.
    pub namespace: String,
    /// Number of entries.
    pub entries: u64,
    /// Total size of the keys and values, in bytes.
    pub bytes: u64,
}

/// Drops the entries table of a database written by an incompatible schema.
///
/// The table is dropped rather than emptied: a schema change can rename or
/// retype a column, and keeping the old table would make every later statement
/// fail instead of costing one cold start.
fn migrate(conn: &Connection) -> Result<(), rusqlite::Error> {
    let found: Option<String> = conn
        .query_row("SELECT v FROM meta WHERE k = 'schema_version'", [], |row| {
            row.get(0)
        })
        .optional()?;

    let expected = SCHEMA_VERSION.to_string();
    if found.as_deref() == Some(expected.as_str()) {
        return Ok(());
    }

    if let Some(found) = found {
        log::warn!(
            "cubecl cache: database schema {found} is not {expected}, discarding cached entries"
        );
    }
    // Also runs when no version is recorded at all: such a file predates the
    // `meta` table, so whatever `entries` it holds cannot be trusted either.
    conn.execute("DROP TABLE IF EXISTS entries", [])?;

    conn.execute(
        "INSERT INTO meta (k, v) VALUES ('schema_version', ?1) \
         ON CONFLICT(k) DO UPDATE SET v = excluded.v",
        params![expected],
    )?;

    Ok(())
}

/// The file system implementation of [`Storage`], one instance per logical
/// namespace, all sharing the cache root's database.
#[derive(Debug)]
pub struct SqliteStorage {
    database: Database,
    namespace: String,
}

impl SqliteStorage {
    /// Binds a storage to one namespace of a database.
    pub fn new(database: Database, namespace: String) -> Self {
        Self {
            database,
            namespace,
        }
    }
}

impl Storage for SqliteStorage {
    fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        self.database.get(&self.namespace, key)
    }

    fn insert(&self, key: &[u8], value: &[u8]) -> Option<Vec<u8>> {
        self.database.insert(&self.namespace, key, value)
    }

    fn scan(&self, visit: &mut dyn FnMut(Vec<u8>, Vec<u8>)) {
        self.database.scan(&self.namespace, visit)
    }

    fn describe(&self) -> String {
        std::format!("{:?} [{}]", self.database.path(), self.namespace)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two independent connections to one file, which is what two processes
    /// sharing a cache root come down to. Exactly one insert may win, and the
    /// loser must be told which value is actually stored.
    #[test_log::test]
    #[cfg_attr(miri, ignore)]
    fn concurrent_connections_agree_on_the_winner() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(DB_FILE_NAME);

        let first = Database::open(&path, false).unwrap();
        let second = Database::open(&path, false).unwrap();

        assert_eq!(first.insert("namespace", b"key", b"first"), None);

        // The second connection sees the committed entry and leaves it alone.
        assert_eq!(
            second.insert("namespace", b"key", b"second"),
            Some(b"first".to_vec())
        );
        assert_eq!(second.get("namespace", b"key"), Some(b"first".to_vec()));
    }

    /// Many writers on one file must block on each other rather than fail:
    /// this is what `busy_timeout` buys, and what the previous lock sidecar
    /// approximated with a sleep loop.
    #[test_log::test]
    #[cfg_attr(miri, ignore)]
    fn concurrent_writers_do_not_lose_entries() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(DB_FILE_NAME);
        // Create the file up front so every writer opens the same schema.
        let database = Database::open(&path, false).unwrap();

        std::thread::scope(|scope| {
            for writer in 0..4 {
                let path = &path;
                scope.spawn(move || {
                    let database = Database::open(path, false).unwrap();
                    for entry in 0..25 {
                        let key = std::format!("{writer}-{entry}");
                        assert_eq!(database.insert("namespace", key.as_bytes(), b"value"), None);
                    }
                });
            }
        });

        let mut count = 0;
        database.scan("namespace", &mut |_key, _value| count += 1);
        assert_eq!(count, 100);
    }

    /// A database written by another schema must be rebuilt, not misread.
    ///
    /// The table is dropped rather than emptied, so a schema that renamed or
    /// retyped a column still recovers. Emptying it would leave the old
    /// columns in place and make every later statement fail forever.
    #[test_log::test]
    #[cfg_attr(miri, ignore)]
    fn an_incompatible_schema_is_rebuilt() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(DB_FILE_NAME);

        // A database from a build whose entries table had different columns.
        let database = Database::open(&path, false).unwrap();
        database.with_connection(|conn| {
            conn.execute("DROP TABLE entries", []).unwrap();
            conn.execute(
                "CREATE TABLE entries (store TEXT NOT NULL, key BLOB NOT NULL, \
                 value BLOB NOT NULL, PRIMARY KEY (store, key))",
                [],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO entries (store, key, value) VALUES ('old', x'00', x'00')",
                [],
            )
            .unwrap();
            conn.execute("UPDATE meta SET v = '999' WHERE k = 'schema_version'", [])
                .unwrap();
        });
        drop(database);

        let database = Database::open(&path, false).unwrap();
        assert_eq!(database.get("old", b"\x00"), None, "stale rows are gone");
        // The rebuilt table must be usable, which an emptied one would not be.
        assert_eq!(database.insert("namespace", b"key", b"value"), None);
        assert_eq!(
            database.get("namespace", b"key"),
            Some(b"value".to_vec()),
            "the rebuilt table accepts the current column layout"
        );
    }
}
