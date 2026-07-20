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

use super::storage::{InsertSummary, Insertion, NamespaceSummary, Origin, Storage, replaces};
use crate::bytes::Bytes;
use crate::sync::{Arc, Lazy, Mutex};

/// File name of the cache database of the environment named `name`.
pub fn db_file_name(name: &str) -> String {
    crate::environment::file_name(name)
}

/// The database schema this build reads and writes. A file carrying any other
/// version has its entries table dropped and rebuilt: it is a cache, so the
/// only cost is one cold start. Bump this on any change to
/// [`CREATE_ENTRIES`], including a renamed column.
pub const SCHEMA_VERSION: u32 = 3;

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
    key    BLOB NOT NULL,
    value  BLOB NOT NULL,
    origin INTEGER NOT NULL,
    PRIMARY KEY (namespace, key)
);
";

const INSERT_SQL: &str = "INSERT INTO entries (namespace, key, value, origin) \
                          VALUES (?1, ?2, ?3, ?4) ON CONFLICT DO NOTHING";
const REPLACE_SQL: &str = "UPDATE entries SET value = ?3, origin = ?4 \
                           WHERE namespace = ?1 AND key = ?2";
const UPSERT_SQL: &str = "INSERT INTO entries (namespace, key, value, origin) \
                          VALUES (?1, ?2, ?3, ?4) \
                          ON CONFLICT (namespace, key) \
                          DO UPDATE SET value = excluded.value, origin = excluded.origin";
const SELECT_SQL: &str = "SELECT value FROM entries WHERE namespace = ?1 AND key = ?2";
const SELECT_WITH_ORIGIN_SQL: &str =
    "SELECT value, origin FROM entries WHERE namespace = ?1 AND key = ?2";
const SCAN_SQL: &str = "SELECT key, value FROM entries WHERE namespace = ?1";

/// `Origin` as stored in the `origin` column.
fn origin_code(origin: Origin) -> i64 {
    match origin {
        Origin::Local => 0,
        Origin::Imported => 1,
    }
}

fn origin_from_code(code: i64) -> Origin {
    match code {
        1 => Origin::Imported,
        _ => Origin::Local,
    }
}

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
    pub fn open_active() -> Option<Self> {
        Self::open_at(crate::environment::root(), &crate::environment::active())
    }

    /// Opens the database of a named environment under `root`.
    pub fn open_at<P: AsRef<Path>>(root: P, environment: &str) -> Option<Self> {
        let root = root.as_ref();
        let path = root.join(db_file_name(environment));

        let mut opened = OPENED.lock();
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
        let conn = if read_only {
            open_read_only(path)?
        } else {
            let flags = OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_CREATE
                | OpenFlags::SQLITE_OPEN_NO_MUTEX;
            Connection::open_with_flags(path, flags)?
        };

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
        let mut guard = self.conn.lock();
        func(&mut guard)
    }

    /// The value stored under `key` in `namespace`.
    pub fn get(&self, namespace: &str, key: &[u8]) -> Option<Bytes> {
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
                .map(Bytes::from_bytes_vec)
        })
    }

    /// Stores `value` under `key`. See [`Storage`] for the rules; in short, a
    /// local value replaces an imported one and nothing else overwrites.
    pub fn insert(&self, namespace: &str, key: &[u8], value: &[u8], origin: Origin) -> Insertion {
        let result = self.with_connection(|conn| {
            let transaction = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
            let outcome = insert_one(&transaction, namespace, key, value, origin)?;
            transaction.commit()?;
            Ok::<_, rusqlite::Error>(outcome)
        });

        self.report("write", result)
    }

    /// Stores every entry of `entries` under the rules of [`insert`](Self::insert),
    /// in a single transaction.
    ///
    /// One exclusive lock for the whole batch: importing a bundle entry by
    /// entry would take one per entry and block every other writer of the
    /// cache root for the duration.
    pub fn insert_many(
        &self,
        namespace: &str,
        entries: &mut dyn Iterator<Item = (Bytes, Bytes)>,
        origin: Origin,
    ) -> InsertSummary {
        let result = self.with_connection(|conn| {
            let transaction = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;

            let mut summary = InsertSummary::default();
            for (key, value) in entries {
                summary.record(&insert_one(&transaction, namespace, &key, &value, origin)?);
            }

            transaction.commit()?;
            Ok::<_, rusqlite::Error>(summary)
        });

        match result {
            Ok(summary) => summary,
            Err(err) => {
                self.warn("batch write", err);
                InsertSummary::default()
            }
        }
    }

    /// Stores `value` under `key`, overwriting whatever is there.
    pub fn replace(&self, namespace: &str, key: &[u8], value: &[u8], origin: Origin) -> Insertion {
        let result = self.with_connection(|conn| {
            conn.prepare_cached(UPSERT_SQL)?.execute(params![
                namespace,
                key,
                value,
                origin_code(origin)
            ])?;
            Ok::<_, rusqlite::Error>(Insertion::Stored)
        });

        self.report("replace", result)
    }

    /// Rewrites the file so it can be read from a directory nobody may write,
    /// then it is ready to ship.
    ///
    /// A WAL database records that mode in its header, and reading one means
    /// creating a `-shm` wal-index beside the file. Ship a bundle that way and
    /// installing it read-only — a container image layer, a Nix store,
    /// `/usr/share` — makes every lookup miss with no error at all. Rolling
    /// the journal back to `DELETE` removes that requirement.
    ///
    /// Call it on a finished file with no other connection open to it.
    pub fn finalize_for_shipping(&self) -> Result<(), rusqlite::Error> {
        self.with_connection(|conn| {
            // `journal_mode` answers with the mode it settled on, which plain
            // `pragma_update` rejects as unexpected rows.
            conn.pragma_update_and_check(None, "journal_mode", "DELETE", |_row| Ok(()))
        })
    }

    /// Turns a failed statement into [`Insertion::Failed`] rather than letting
    /// it pass for a successful write.
    fn report(&self, operation: &str, result: Result<Insertion, rusqlite::Error>) -> Insertion {
        match result {
            Ok(insertion) => insertion,
            Err(err) => {
                let message = err.to_string();
                self.warn(operation, err);
                Insertion::Failed(message)
            }
        }
    }

    /// Visits every entry of `namespace`.
    pub fn scan(&self, namespace: &str, visit: &mut dyn FnMut(&[u8], &[u8])) {
        let result = self.with_connection(|conn| {
            let mut stmt = conn.prepare_cached(SCAN_SQL)?;
            let mut rows = stmt.query(params![namespace])?;
            while let Some(row) = rows.next()? {
                // `get_ref` borrows from the statement, so a scan of a large
                // namespace doesn't allocate a Vec per column.
                visit(row.get_ref(0)?.as_blob()?, row.get_ref(1)?.as_blob()?);
            }
            Ok::<_, rusqlite::Error>(())
        });

        if let Err(err) = result {
            self.warn("scan", err);
        }
    }

    /// Every namespace this database holds, with its entry count and size.
    ///
    /// This is what a cache root offers for bundling.
    pub fn namespaces(&self) -> Vec<String> {
        self.summary()
            .into_iter()
            .map(|summary| summary.namespace)
            .collect()
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

/// One insert inside an already-open transaction, shared by the single and
/// batch paths so both arbitrate collisions the same way.
fn insert_one(
    conn: &Connection,
    namespace: &str,
    key: &[u8],
    value: &[u8],
    origin: Origin,
) -> Result<Insertion, rusqlite::Error> {
    let code = origin_code(origin);

    let written = conn
        .prepare_cached(INSERT_SQL)?
        .execute(params![namespace, key, value, code])?;

    if written != 0 {
        return Ok(Insertion::Stored);
    }

    let found: Option<(Vec<u8>, i64)> = conn
        .prepare_cached(SELECT_WITH_ORIGIN_SQL)?
        .query_row(params![namespace, key], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .optional()?;

    match found {
        Some((_, existing_code)) if replaces(origin, origin_from_code(existing_code)) => {
            conn.prepare_cached(REPLACE_SQL)?
                .execute(params![namespace, key, value, code])?;
            Ok(Insertion::Stored)
        }
        Some((existing, _)) => Ok(Insertion::Conflict(Bytes::from_bytes_vec(existing))),
        // The row vanished between the insert and the read, which only a
        // concurrent schema rebuild can do. Treat it as a lost write.
        None => Ok(Insertion::Failed(String::from(
            "the entry disappeared during the write",
        ))),
    }
}

/// Opens a database file for reading, tolerating a directory we may not write.
///
/// WAL mode is recorded in the file header, and a reader of a WAL database has
/// to create a `-shm` wal-index *next to the file*. A bundle installed in a
/// container image layer, a Nix store or `/usr/share` therefore fails to open
/// at all, which surfaces as zero imported entries and no error. The URI
/// `immutable=1` form skips locking and the wal-index entirely, which is
/// exactly right for a file nobody can write — and wrong for a cache another
/// process is appending to, so it is only ever a fallback.
fn open_read_only(path: &Path) -> Result<Connection, rusqlite::Error> {
    let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX;

    let attempt = Connection::open_with_flags(path, flags).and_then(|conn| {
        // Opening is lazy; reading the schema is what forces the wal-index to
        // be created, so a successful open alone proves nothing.
        conn.query_row("SELECT count(*) FROM sqlite_schema", [], |_| Ok(()))?;
        Ok(conn)
    });

    let err = match attempt {
        Ok(conn) => return Ok(conn),
        Err(err) => err,
    };

    let Some(uri) = immutable_uri(path) else {
        return Err(err);
    };

    log::debug!("cubecl cache: {path:?} is not readable in place ({err}); retrying immutable");
    Connection::open_with_flags(uri, flags | OpenFlags::SQLITE_OPEN_URI)
}

/// The `file:` URI addressing `path` as an immutable database, or `None` for a
/// path `SQLite` can't be given as a URI.
fn immutable_uri(path: &Path) -> Option<String> {
    let path = path.to_str()?;

    let mut uri = String::from("file:");
    for character in path.chars() {
        match character {
            // The only characters that would end the path component of a URI.
            '?' => uri.push_str("%3f"),
            '#' => uri.push_str("%23"),
            '%' => uri.push_str("%25"),
            _ => uri.push(character),
        }
    }
    uri.push_str("?immutable=1");

    Some(uri)
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
    fn get(&self, key: &[u8]) -> Option<Bytes> {
        self.database.get(&self.namespace, key)
    }

    fn insert(&self, key: &[u8], value: Bytes, origin: Origin) -> Insertion {
        self.database.insert(&self.namespace, key, &value, origin)
    }

    fn replace(&self, key: &[u8], value: Bytes, origin: Origin) -> Insertion {
        self.database.replace(&self.namespace, key, &value, origin)
    }

    fn insert_many(
        &self,
        entries: &mut dyn Iterator<Item = (Bytes, Bytes)>,
        origin: Origin,
    ) -> InsertSummary {
        self.database.insert_many(&self.namespace, entries, origin)
    }

    fn scan(&self, visit: &mut dyn FnMut(&[u8], &[u8])) {
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
        let path = dir.path().join(db_file_name("test"));

        let first = Database::open(&path, false).unwrap();
        let second = Database::open(&path, false).unwrap();

        assert_eq!(
            first.insert("namespace", b"key", b"first", Origin::Local),
            Insertion::Stored
        );

        // The second connection sees the committed entry and leaves it alone.
        assert_eq!(
            second.insert("namespace", b"key", b"second", Origin::Local),
            Insertion::Conflict(Bytes::from_bytes_vec(b"first".to_vec()))
        );
        assert_eq!(
            second.get("namespace", b"key").as_deref(),
            Some(&b"first"[..])
        );
    }

    /// Many writers on one file must block on each other rather than fail:
    /// this is what `busy_timeout` buys, and what the previous lock sidecar
    /// approximated with a sleep loop.
    #[test_log::test]
    #[cfg_attr(miri, ignore)]
    fn concurrent_writers_do_not_lose_entries() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(db_file_name("test"));
        // Create the file up front so every writer opens the same schema.
        let database = Database::open(&path, false).unwrap();

        std::thread::scope(|scope| {
            for writer in 0..4 {
                let path = &path;
                scope.spawn(move || {
                    let database = Database::open(path, false).unwrap();
                    for entry in 0..25 {
                        let key = std::format!("{writer}-{entry}");
                        assert_eq!(
                            database.insert("namespace", key.as_bytes(), b"value", Origin::Local),
                            Insertion::Stored
                        );
                    }
                });
            }
        });

        let mut count = 0;
        database.scan("namespace", &mut |_key, _value| count += 1);
        assert_eq!(count, 100);
    }

    /// A bundle ships in a directory the application may not write: a
    /// container image layer, a Nix store, `/usr/share`. Reading a WAL
    /// database there means creating a `-shm` wal-index beside it, which
    /// fails, and a failed read is indistinguishable from an empty cache.
    #[test_log::test]
    #[cfg(unix)]
    #[cfg_attr(miri, ignore)]
    fn a_database_in_a_read_only_directory_is_readable() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(db_file_name("test"));

        let database = Database::open(&path, false).unwrap();
        database.insert("namespace", b"key", b"value", Origin::Local);
        drop(database);

        let restore = std::fs::metadata(dir.path()).unwrap().permissions();
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o555)).unwrap();

        // Root ignores the mode bits, so the test would prove nothing there.
        let writable = std::fs::File::create(dir.path().join("probe")).is_ok();

        let read = (!writable).then(|| {
            let database = Database::open(&path, true).expect("read-only open");
            database.get("namespace", b"key")
        });

        std::fs::set_permissions(dir.path(), restore).unwrap();

        if let Some(read) = read {
            assert_eq!(read.as_deref(), Some(&b"value"[..]));
        }
    }

    /// The other half of the same problem: a file about to be shipped should
    /// not carry WAL in its header at all.
    #[test_log::test]
    #[cfg_attr(miri, ignore)]
    fn finalizing_clears_the_wal_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(db_file_name("test"));

        let database = Database::open(&path, false).unwrap();
        database.insert("namespace", b"key", b"value", Origin::Local);
        database.finalize_for_shipping().unwrap();

        let mode: String = database.with_connection(|conn| {
            conn.query_row("PRAGMA journal_mode", [], |row| row.get(0))
                .unwrap()
        });
        assert_eq!(mode, "delete");
        assert!(!path.with_extension("db-wal").exists());
    }

    /// An entry whose bytes no longer decode must be repairable: `insert`
    /// refuses to overwrite it, and without `replace` the key would be wedged
    /// for the life of the file.
    #[test_log::test]
    #[cfg_attr(miri, ignore)]
    fn replace_overwrites_an_existing_entry() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(db_file_name("test"));
        let database = Database::open(&path, false).unwrap();

        database.insert("namespace", b"key", b"corrupt", Origin::Local);
        assert_eq!(
            database.insert("namespace", b"key", b"value", Origin::Local),
            Insertion::Conflict(Bytes::from_bytes_vec(b"corrupt".to_vec()))
        );

        assert_eq!(
            database.replace("namespace", b"key", b"value", Origin::Local),
            Insertion::Stored
        );
        assert_eq!(
            database.get("namespace", b"key").as_deref(),
            Some(&b"value"[..])
        );
    }

    /// A batch arbitrates collisions exactly like the entries would one by
    /// one, it just takes one lock instead of N.
    #[test_log::test]
    #[cfg_attr(miri, ignore)]
    fn insert_many_matches_insert() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(db_file_name("test"));
        let database = Database::open(&path, false).unwrap();

        database.insert("namespace", b"taken", b"local", Origin::Local);

        let bytes = |value: &[u8]| Bytes::from_bytes_vec(value.to_vec());
        let entries = std::vec![
            (bytes(b"fresh"), bytes(b"imported")),
            (bytes(b"taken"), bytes(b"imported")),
        ];

        let summary = database.insert_many("namespace", &mut entries.into_iter(), Origin::Imported);
        assert_eq!(summary.stored, 1);
        assert_eq!(summary.conflict, 1);
        assert_eq!(summary.failed, 0);

        assert_eq!(
            database.get("namespace", b"fresh").as_deref(),
            Some(&b"imported"[..])
        );
        assert_eq!(
            database.get("namespace", b"taken").as_deref(),
            Some(&b"local"[..]),
            "a bundle never overwrites a local value"
        );
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
        let path = dir.path().join(db_file_name("test"));

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
        assert_eq!(
            database.insert("namespace", b"key", b"value", Origin::Local),
            Insertion::Stored
        );
        assert_eq!(
            database.get("namespace", b"key").as_deref(),
            Some(&b"value"[..]),
            "the rebuilt table accepts the current column layout"
        );
    }
}
