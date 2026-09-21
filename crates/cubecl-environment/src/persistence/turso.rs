//! Turso persistence: the database file shared by every namespace of an
//! environment.
//!
//! The engine's API is a set of futures, but its work is synchronous: a
//! statement steps, asks its I/O for a page, and steps again once the I/O has
//! answered. Natively the file system answers within the call, and the
//! browser's files are synchronous access handles that do the same, so a
//! statement never waits on anything an event loop would deliver. Every
//! statement here is driven to completion on the spot ([`drive`]), and the
//! [`Storage`] this module hands out is synchronous like every other.
//!
//! Opening is the exception: the browser reaches its files through promises.
//! The database therefore opens through an `async` step ([`open_ahead`]),
//! which a page awaits once before anything needs it; natively a storage
//! opens the database itself on first use, blocking on the file system as a
//! file read would.

use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::future::Future;

use hashbrown::HashMap;
use turso::transaction::TransactionBehavior;

use super::{InsertSummary, Insertion, NamespaceSummary, Origin, Storage};
use crate::bytes::Bytes;
use crate::sync::{LazyLock, Mutex};

/// The database schema this build reads and writes. A file carrying any other
/// version has its entries table dropped and rebuilt: it is a cache, so the
/// only cost is one cold start. Bump this on any change to
/// [`CREATE_ENTRIES`], including a renamed column.
pub const SCHEMA_VERSION: u32 = 3;

/// The `meta` key holding [`SCHEMA_VERSION`].
pub(crate) const SCHEMA_VERSION_KEY: &str = "schema_version";

/// Created first and never dropped, so the schema version survives a rebuild
/// of the entries table.
const CREATE_META: &str = "
    CREATE TABLE IF NOT EXISTS meta (
        k TEXT PRIMARY KEY,
        v TEXT NOT NULL
    )
";

const META_GET: &str = "SELECT v FROM meta WHERE k = ?1";

const META_SET: &str = "INSERT INTO meta (k, v) VALUES (?1, ?2) \
                        ON CONFLICT(k) DO UPDATE SET v = excluded.v";

const CREATE_ENTRIES: &str = "
    CREATE TABLE IF NOT EXISTS entries (
        namespace TEXT NOT NULL,
        key BLOB NOT NULL,
        value BLOB NOT NULL,
        origin INTEGER NOT NULL,
        PRIMARY KEY (namespace, key)
    )
";

const DROP_ENTRIES: &str = "DROP TABLE IF EXISTS entries";

/// The [`Storage`] insert rule in one statement: insert-only, except that a
/// local value (origin 0) replaces an imported one (origin 1). The primary
/// key arbitrates, so the check and the write are one atomic step, and the
/// count of changed rows says whether anything was stored.
const INSERT: &str = "INSERT INTO entries (namespace, key, value, origin) \
                      VALUES (?1, ?2, ?3, ?4) \
                      ON CONFLICT(namespace, key) DO UPDATE \
                      SET value = excluded.value, origin = excluded.origin \
                      WHERE entries.origin = 1 AND excluded.origin = 0";

const REPLACE: &str = "INSERT INTO entries (namespace, key, value, origin) \
                       VALUES (?1, ?2, ?3, ?4) \
                       ON CONFLICT(namespace, key) DO UPDATE \
                       SET value = excluded.value, origin = excluded.origin";

const SELECT: &str = "SELECT value FROM entries WHERE namespace = ?1 AND key = ?2";

const SCAN: &str = "SELECT key, value FROM entries WHERE namespace = ?1";

const PURGE: &str = "DELETE FROM entries WHERE namespace = ?1";

const PURGE_KEY: &str = "DELETE FROM entries WHERE namespace = ?1 AND key = ?2";

const SUMMARY: &str = "SELECT namespace, COUNT(*), SUM(length(key) + length(value)) \
                       FROM entries GROUP BY namespace ORDER BY namespace";

type DatabaseResult = Result<Arc<turso::Database>, String>;

enum DatabaseState {
    Opening(Vec<async_channel::Sender<DatabaseResult>>),
    Ready(Arc<turso::Database>),
}

/// The databases this process has opened, by location, so that every
/// namespace of an environment shares one.
static DATABASES: LazyLock<Mutex<HashMap<String, DatabaseState>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Every namespace of the active environment's database at once.
///
/// A [`TursoStorage`] is this bound to one namespace. The records of a build
/// ([`crate::records`]) span namespaces — the sessions', and one per record
/// kind — and are read and written through this instead.
pub struct Database {
    /// The engine rejects concurrent use of a connection at runtime rather
    /// than serializing it, so each handle holds one and takes it in turn.
    connection: Mutex<turso::Connection>,
    location: String,
}

impl core::fmt::Debug for Database {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("Database")
            .field("location", &self.location)
            .finish()
    }
}

impl Database {
    /// A connection to the active environment's database.
    fn open() -> Result<Self, String> {
        let location = location()?;
        let database = database()?;
        let connection = connect(&database).map_err(error)?;

        Ok(Self {
            connection: Mutex::new(connection),
            location,
        })
    }

    /// The active environment's database, or `None` when it can't be opened:
    /// a read-only mount, a missing parent directory, a browser page that
    /// hasn't awaited [`crate::environment::open`]. Callers go without
    /// rather than fail.
    pub fn open_active() -> Option<Self> {
        match Self::open() {
            Ok(database) => Some(database),
            Err(error) => {
                log::warn!("Unable to open the Turso cache: {error}");
                None
            }
        }
    }

    /// Runs `operation` on the connection, logging a failure under `name`
    /// against `namespace` and reporting it as a message.
    fn run<T>(
        &self,
        name: &str,
        namespace: &str,
        operation: impl FnOnce(&mut turso::Connection) -> Result<T, turso::Error>,
    ) -> Result<T, String> {
        operation(&mut self.connection.lock()).map_err(|err| {
            log::warn!(
                "Unable to {name} {}: {err}",
                describe(&self.location, namespace)
            );
            err.to_string()
        })
    }

    /// The value stored under `key` in `namespace`.
    pub fn get(&self, namespace: &str, key: &[u8]) -> Option<Bytes> {
        self.run("read", namespace, |connection| {
            let mut rows = drive(connection.query(SELECT, (namespace, key.to_vec())))?;
            let Some(row) = drive(rows.next())? else {
                return Ok(None);
            };
            let value: Vec<u8> = row.get(0)?;
            Ok(Some(Bytes::from_bytes_vec(value)))
        })
        .unwrap_or_default()
    }

    /// Stores `value` under `key`. See [`Storage`] for the rules; in short, a
    /// local value replaces an imported one and nothing else overwrites.
    pub fn insert(&self, namespace: &str, key: &[u8], value: &[u8], origin: Origin) -> Insertion {
        self.run("write", namespace, |connection| {
            insert_on(connection, namespace, key, value, origin)
        })
        .unwrap_or_else(Insertion::Failed)
    }

    /// Stores `value` under `key`, overwriting whatever is there.
    pub fn replace(&self, namespace: &str, key: &[u8], value: &[u8], origin: Origin) -> Insertion {
        self.run("replace", namespace, |connection| {
            let params = (namespace, key.to_vec(), value.to_vec(), origin_code(origin));
            drive(connection.execute(REPLACE, params))?;
            Ok(Insertion::Stored)
        })
        .unwrap_or_else(Insertion::Failed)
    }

    /// Stores every entry of `entries` under the rules of
    /// [`insert`](Self::insert), in one transaction: it holds the writer once
    /// rather than once per entry, and lands as a whole.
    pub fn insert_many(
        &self,
        namespace: &str,
        entries: &mut dyn Iterator<Item = (Bytes, Bytes)>,
        origin: Origin,
    ) -> InsertSummary {
        let mut connection = self.connection.lock();
        let transaction = match drive(connection.transaction_with_behavior(write_transaction())) {
            Ok(transaction) => transaction,
            Err(err) => {
                log::warn!(
                    "Unable to batch write {}: {err}",
                    describe(&self.location, namespace)
                );
                return InsertSummary {
                    failed: entries.count(),
                    ..InsertSummary::default()
                };
            }
        };

        let mut summary = InsertSummary::default();
        for (key, value) in entries {
            match insert_on(&transaction, namespace, &key, &value, origin) {
                Ok(insertion) => summary.record(&insertion),
                Err(_) => summary.failed += 1,
            }
        }
        if let Err(err) = drive(transaction.commit()) {
            log::warn!(
                "Unable to commit a batch write to {}: {err}",
                describe(&self.location, namespace)
            );
            summary.failed += summary.stored;
            summary.stored = 0;
        }
        summary
    }

    /// Visits every entry of `namespace`.
    pub fn scan(&self, namespace: &str, visit: &mut dyn FnMut(&[u8], &[u8])) {
        let _ = self.run("scan", namespace, |connection| {
            let mut rows = drive(connection.query(SCAN, (namespace,)))?;
            while let Some(row) = drive(rows.next())? {
                let key: Vec<u8> = row.get(0)?;
                let value: Vec<u8> = row.get(1)?;
                visit(&key, &value);
            }
            Ok(())
        });
    }

    /// Deletes every entry of `namespace`. Logs a failed delete rather than
    /// reporting it; see the [`Storage`] contract.
    pub fn purge(&self, namespace: &str) {
        let _ = self.run("purge", namespace, |connection| {
            drive(connection.execute(PURGE, (namespace,))).map(|_| ())
        });
    }

    /// Deletes the entry of `namespace` under `key`, with the same failure
    /// contract as [`purge`](Self::purge).
    pub fn purge_key(&self, namespace: &str, key: &[u8]) {
        let _ = self.run("purge a key from", namespace, |connection| {
            drive(connection.execute(PURGE_KEY, (namespace, key.to_vec()))).map(|_| ())
        });
    }

    /// The names of every namespace this database holds.
    pub fn namespaces(&self) -> Vec<String> {
        self.run("summarize", "", |connection| summarize(connection))
            .unwrap_or_default()
            .into_iter()
            .map(|summary| summary.namespace)
            .collect()
    }
}

/// One insert on `connection`: the statement arbitrates, and only a declined
/// write costs a second one, to fetch the value that won.
fn insert_on(
    connection: &turso::Connection,
    namespace: &str,
    key: &[u8],
    value: &[u8],
    origin: Origin,
) -> Result<Insertion, turso::Error> {
    let params = (namespace, key.to_vec(), value.to_vec(), origin_code(origin));
    if drive(connection.execute(INSERT, params))? == 1 {
        return Ok(Insertion::Stored);
    }

    let mut rows = drive(connection.query(SELECT, (namespace, key.to_vec())))?;
    match drive(rows.next())? {
        Some(row) => Ok(Insertion::Conflict(Bytes::from_bytes_vec(row.get(0)?))),
        // Purged between the two statements: nothing to report as the
        // winner, and nothing stored.
        None => Ok(Insertion::Failed(
            "the entry was removed while it was being written".to_string(),
        )),
    }
}

fn describe(location: &str, namespace: &str) -> String {
    format!("Turso {location} ({namespace})")
}

/// One namespace of the active environment's database.
pub struct TursoStorage {
    database: Database,
    namespace: String,
}

impl core::fmt::Debug for TursoStorage {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("TursoStorage")
            .field("namespace", &self.namespace)
            .field("location", &self.database.location)
            .finish()
    }
}

impl TursoStorage {
    /// Binds a storage to `namespace` in the active environment's database.
    pub fn open(namespace: String) -> Result<Self, String> {
        Ok(Self {
            database: Database::open()?,
            namespace,
        })
    }
}

impl Storage for TursoStorage {
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
        self.database.scan(&self.namespace, visit);
    }

    fn purge(&self) {
        self.database.purge(&self.namespace);
    }

    fn purge_key(&self, key: &[u8]) {
        self.database.purge_key(&self.namespace, key);
    }

    fn describe(&self) -> String {
        describe(&self.database.location, &self.namespace)
    }
}

/// Runs one of the engine's futures to completion.
///
/// Natively this blocks the thread, as reading a file would. In the browser
/// nothing can block, and nothing has to: an engine future is pending only
/// between a step that asked for I/O and the next, and the browser's files
/// ([`super::turso_browser`]) answer within the call, so the next poll
/// finds the answer. Polling in a loop is therefore how it is driven — and
/// why a future that awaits the event loop must never come through here.
/// Opening the database does, and stays `async`.
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

/// The active environment's database.
///
/// Natively a database that isn't open yet is opened here, blocking on the
/// file system. The browser can't block on its files' promises: there the
/// database must have been opened ahead ([`open_ahead`]), and a storage that
/// finds it closed is told so, and falls back to memory.
pub(crate) fn database() -> DatabaseResult {
    let location = location()?;
    if let Some(DatabaseState::Ready(database)) = DATABASES.lock().get(&location) {
        return Ok(database.clone());
    }

    #[cfg(native_cache)]
    {
        crate::future::block_on(shared_database(&location))
    }

    #[cfg(browser_cache)]
    {
        Err(format!(
            "the database of environment '{}' is not open; await `environment::open()` \
             before the first use of its caches",
            crate::environment::active()
        ))
    }
}

/// Opens the active environment's database, from a place that can await.
pub(crate) async fn open_ahead() -> Result<(), String> {
    shared_database(&location()?).await.map(|_| ())
}

async fn shared_database(location: &str) -> DatabaseResult {
    let receiver = {
        let mut databases = DATABASES.lock();
        match databases.get_mut(location) {
            Some(DatabaseState::Ready(database)) => return Ok(database.clone()),
            Some(DatabaseState::Opening(waiters)) => {
                let (sender, receiver) = async_channel::bounded(1);
                waiters.push(sender);
                Some(receiver)
            }
            None => {
                databases.insert(location.to_string(), DatabaseState::Opening(Vec::new()));
                None
            }
        }
    };

    if let Some(receiver) = receiver {
        return receiver
            .recv()
            .await
            .map_err(|error| format!("database initialization was cancelled: {error}"))?;
    }

    // Whatever happens to this future — including being dropped halfway, by
    // a timeout or a panic — the registry is settled and the waiters told.
    let mut opener = Opener {
        location,
        result: Err("database initialization was cancelled".to_string()),
    };

    let writable = match open_database(location).await {
        Ok(database) => migrate(&database).map(|()| database),
        Err(err) => Err(err),
    };

    opener.result = match writable {
        Ok(database) => Ok(Arc::new(database)),
        // A lock another process holds is not a read-only location. Report
        // it and cache nothing, so the next open tries the writable path
        // again rather than serving a read-only file for the rest of the
        // process.
        Err(err @ (turso::Error::Busy(_) | turso::Error::BusySnapshot(_))) => Err(error(err)),
        #[cfg(native_cache)]
        Err(err) => open_read_only(location, &err.to_string())
            .await
            .map(Arc::new),
        #[cfg(not(native_cache))]
        Err(err) => Err(error(err)),
    };
    opener.result.clone()
}

/// The registry's `Opening` entry for one location, settled when the opener
/// is dropped: replaced by `Ready` on success, removed otherwise, and every
/// waiter handed the outcome either way.
struct Opener<'a> {
    location: &'a str,
    result: DatabaseResult,
}

impl Drop for Opener<'_> {
    fn drop(&mut self) {
        let waiters = {
            let mut databases = DATABASES.lock();
            let waiters = match databases.remove(self.location) {
                Some(DatabaseState::Opening(waiters)) => waiters,
                _ => Vec::new(),
            };
            if let Ok(database) = &self.result {
                databases.insert(
                    self.location.to_string(),
                    DatabaseState::Ready(database.clone()),
                );
            }
            waiters
        };

        for waiter in waiters {
            let _ = waiter.try_send(self.result.clone());
        }
    }
}

#[cfg(native_cache)]
async fn open_database(location: &str) -> Result<turso::Database, turso::Error> {
    if let Some(parent) = std::path::Path::new(location).parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| turso::Error::IoError(error.kind(), "creating the cache directory"))?;
    }

    turso::Builder::new_local(location)
        .experimental_multiprocess_wal(true)
        .build()
        .await
}

/// A database nobody may write, served as it is: a cache root in a container
/// image layer, a Nix store path, a mounted bundle.
///
/// Only a file already at this build's schema qualifies. Anything else would
/// need the rebuild [`migrate`] performs, which needs a writable file; the
/// caller falls back to memory instead.
#[cfg(native_cache)]
async fn open_read_only(location: &str, err: &str) -> Result<turso::Database, String> {
    log::debug!("cubecl cache: {location} is not writable ({err}); opening read-only");

    let database = turso::Builder::new_local(location)
        .read_only(true)
        .build()
        .await
        .map_err(error)?;
    let connection = connect(&database).map_err(error)?;

    let expected = SCHEMA_VERSION.to_string();
    match meta_get(&connection, SCHEMA_VERSION_KEY).map_err(error)? {
        Some(found) if found == expected => Ok(database),
        found => Err(format!(
            "read-only database at {location} has schema {found:?}, expected {expected}"
        )),
    }
}

#[cfg(browser_cache)]
async fn open_database(location: &str) -> Result<turso::Database, turso::Error> {
    let wal = format!("{location}-wal");
    let io = super::turso_browser::BrowserIo::new(&[location, &wal])
        .await
        .map_err(turso::Error::Error)?;
    turso::Builder::new_local(location)
        .with_io_impl(Arc::new(io))
        .build()
        .await
}

#[cfg(native_cache)]
fn location() -> Result<String, String> {
    crate::environment::path()
        .to_str()
        .map(ToString::to_string)
        .ok_or_else(|| "cache path is not valid UTF-8".to_string())
}

#[cfg(browser_cache)]
fn location() -> Result<String, String> {
    Ok(format!("cubecl-{}.db", crate::environment::active()))
}

/// The transaction a write takes. Natively `IMMEDIATE`: the writer is
/// taken up front, so two processes opening one file wait on each other
/// rather than fail halfway through. The browser is one tab per environment
/// with nothing to wait for, and could not take one anyway: an immediate
/// transaction opens the engine's temp database, whose clock has no
/// implementation on wasm.
fn write_transaction() -> TransactionBehavior {
    #[cfg(browser_cache)]
    {
        TransactionBehavior::Deferred
    }
    #[cfg(not(browser_cache))]
    {
        TransactionBehavior::Immediate
    }
}

/// Brings the file to [`SCHEMA_VERSION`], dropping the entries of any other.
///
/// The table is dropped rather than emptied: a schema change can rename or
/// retype a column, and keeping the old table would make every later statement
/// fail instead of costing one cold start. The version is read and the schema
/// rebuilt under one write lock, so two processes opening the same file at
/// once rebuild it once: the second waits, then reads the version the first
/// wrote.
pub(crate) fn migrate(database: &turso::Database) -> Result<(), turso::Error> {
    let mut connection = connect(database)?;
    drive(connection.execute(CREATE_META, ()))?;

    let transaction = drive(connection.transaction_with_behavior(write_transaction()))?;

    let expected = SCHEMA_VERSION.to_string();
    let found = meta_get(&transaction, SCHEMA_VERSION_KEY)?;

    if found.as_deref() != Some(expected.as_str()) {
        match &found {
            Some(found) => log::warn!(
                "cubecl cache: database schema {found} is not {expected}, discarding cached entries"
            ),
            // No version at all: the file predates the `meta` table, so
            // whatever entries it holds cannot be trusted either.
            None => log::debug!("cubecl cache: initializing database schema {expected}"),
        }
        drive(transaction.execute(DROP_ENTRIES, ()))?;
        meta_set(&transaction, SCHEMA_VERSION_KEY, &expected)?;
    }

    drive(transaction.execute(CREATE_ENTRIES, ()))?;
    drive(transaction.commit())
}

/// Reads a `meta` row, or `None` when the key is absent.
///
/// This module owns the table, so everything that touches it goes through
/// here: the schema version, and a bundle's manifest.
pub(crate) fn meta_get(
    connection: &turso::Connection,
    key: &str,
) -> Result<Option<String>, turso::Error> {
    let mut rows = drive(connection.query(META_GET, (key,)))?;
    match drive(rows.next())? {
        Some(row) => Ok(Some(row.get(0)?)),
        None => Ok(None),
    }
}

/// Writes a `meta` row, replacing the key's previous value.
pub(crate) fn meta_set(
    connection: &turso::Connection,
    key: &str,
    value: &str,
) -> Result<(), turso::Error> {
    drive(connection.execute(META_SET, (key, value)))?;
    Ok(())
}

/// Entry count and total size per namespace of the active environment's
/// database, for reporting. Empty when the database isn't open.
pub(crate) fn summary() -> Vec<NamespaceSummary> {
    let result = database().and_then(|database| {
        let connection = connect(&database).map_err(error)?;
        summarize(&connection).map_err(error)
    });

    result.unwrap_or_else(|err| {
        log::warn!("Unable to summarize the cache: {err}");
        Vec::new()
    })
}

/// Entry count and total size per namespace of the database behind
/// `connection`.
pub(crate) fn summarize(
    connection: &turso::Connection,
) -> Result<Vec<NamespaceSummary>, turso::Error> {
    let mut rows = drive(connection.query(SUMMARY, ()))?;
    let mut summaries = Vec::new();
    while let Some(row) = drive(rows.next())? {
        summaries.push(NamespaceSummary {
            namespace: row.get(0)?,
            entries: row.get::<i64>(1)? as u64,
            bytes: row.get::<i64>(2)? as u64,
        });
    }
    Ok(summaries)
}

/// Folds the WAL into the main file and truncates it, so the file stands on
/// its own. Turso never checkpoints on close: a database file copied without
/// its `-wal` is a database missing every write since the last checkpoint.
///
/// Reports whether the checkpoint completed; it doesn't when another
/// connection holds the WAL, and Turso folds every other cause into the same
/// flag while logging the reason itself.
#[cfg(native_cache)]
pub(crate) fn checkpoint(connection: &turso::Connection) -> Result<bool, turso::Error> {
    // The closure's error type is the SDK's, not this crate's; a row that
    // doesn't decode is read as "did not complete" rather than converted.
    let mut incomplete = false;
    drive(connection.pragma_query("wal_checkpoint(TRUNCATE)", |row| {
        incomplete |= row.get::<i64>(0).map_or(true, |busy| busy != 0);
        Ok(())
    }))?;
    Ok(!incomplete)
}

/// How long a statement waits on a lock another process holds before it
/// reports [`Insertion::Failed`]. Several processes sharing a cache root is
/// routine; the wait is yield-based, so it costs the browser nothing it
/// can't afford.
const BUSY_TIMEOUT: core::time::Duration = core::time::Duration::from_secs(5);

/// A connection to `database`, with the busy timeout and `synchronous` set.
///
/// `synchronous` is per-connection and Turso defaults it to `FULL`, which
/// fsyncs on every commit. `NORMAL` may lose the last commits on a power
/// cut, which for a cache costs a recompute — the same trade the rusqlite
/// backend made. It is set on read-only connections too, where it is a no-op,
/// so that every connection this module hands out is configured alike.
pub(crate) fn connect(database: &turso::Database) -> Result<turso::Connection, turso::Error> {
    let connection = database.connect()?;
    connection.busy_timeout(BUSY_TIMEOUT)?;
    // A `PRAGMA` that assigns answers with no rows, which `execute` reports as
    // `Misuse`; `pragma_query` takes it either way.
    drive(connection.pragma_query("synchronous = NORMAL", |_| Ok(())))?;
    Ok(connection)
}

fn origin_code(origin: Origin) -> i64 {
    match origin {
        Origin::Local => 0,
        Origin::Imported => 1,
    }
}

fn error(error: turso::Error) -> String {
    error.to_string()
}

/// The storage serving `namespace` in the active environment's database.
pub fn open(namespace: &str) -> Result<Box<dyn Storage>, String> {
    TursoStorage::open(namespace.to_string()).map(|storage| Box::new(storage) as Box<dyn Storage>)
}

#[cfg(all(test, native_cache))]
mod tests {
    use super::*;
    use crate::future::block_on;
    use alloc::vec;

    /// The tables a database file holds, by name.
    fn tables(location: &str) -> Vec<String> {
        let database = block_on(open_database(location)).unwrap();
        let connection = connect(&database).unwrap();
        let mut rows = drive(connection.query(
            "SELECT name FROM sqlite_schema WHERE type = 'table' ORDER BY name",
            (),
        ))
        .unwrap();

        let mut names = Vec::new();
        while let Some(row) = drive(rows.next()).unwrap() {
            names.push(row.get::<String>(0).unwrap());
        }
        names
    }

    /// The database file of the active environment, as the storage locates it.
    fn active_location(root: &std::path::Path) -> String {
        crate::environment::set_root(root);
        location().unwrap()
    }

    /// A database written by another schema must be rebuilt, not misread.
    ///
    /// The table is dropped rather than emptied, so a schema that renamed or
    /// retyped a column still recovers. Emptying it would leave the old
    /// columns in place and make every later statement fail forever.
    #[test_log::test]
    #[serial_test::serial]
    #[cfg_attr(miri, ignore)]
    fn an_incompatible_schema_is_rebuilt() {
        let dir = tempfile::tempdir().unwrap();
        let location = active_location(dir.path());

        {
            let database = block_on(open_database(&location)).unwrap();
            let connection = connect(&database).unwrap();
            drive(connection.execute(CREATE_META, ())).unwrap();
            drive(connection.execute(META_SET, (SCHEMA_VERSION_KEY, "999"))).unwrap();
            drive(connection.execute(
                "CREATE TABLE entries (store TEXT NOT NULL, key BLOB NOT NULL, \
                 value BLOB NOT NULL, PRIMARY KEY (store, key))",
                (),
            ))
            .unwrap();
            drive(connection.execute("INSERT INTO entries VALUES ('old', X'01', X'02')", ()))
                .unwrap();
        }

        let storage = TursoStorage::open("old".to_string()).unwrap();
        assert_eq!(storage.get(b"\x01"), None, "stale rows are gone");
        // The rebuilt table must be usable, which an emptied one would not be.
        assert_eq!(
            storage.insert(b"key", Bytes::from_bytes_vec(vec![1]), Origin::Local),
            Insertion::Stored,
            "the rebuilt table accepts the current column layout"
        );

        assert_eq!(tables(&location), vec!["entries", "meta"]);

        let database = block_on(open_database(&location)).unwrap();
        let connection = connect(&database).unwrap();
        assert_eq!(
            meta_get(&connection, SCHEMA_VERSION_KEY).unwrap(),
            Some(SCHEMA_VERSION.to_string())
        );
    }

    /// An opener dropped before it finished — a timeout, a panic — must
    /// settle the registry: the location is free to open again, and whoever
    /// was waiting on it is told rather than left waiting forever.
    #[test_log::test]
    #[serial_test::serial]
    fn a_cancelled_open_releases_the_location() {
        let location = "cancelled.db";
        let (sender, receiver) = async_channel::bounded(1);
        DATABASES
            .lock()
            .insert(location.to_string(), DatabaseState::Opening(vec![sender]));

        drop(Opener {
            location,
            result: Err("database initialization was cancelled".to_string()),
        });

        assert!(!DATABASES.lock().contains_key(location));
        assert!(block_on(receiver.recv()).unwrap().is_err());
    }

    /// The version is written once and survives reopening; entries do too,
    /// because a file already at this version is not rebuilt.
    #[test_log::test]
    #[serial_test::serial]
    #[cfg_attr(miri, ignore)]
    fn a_current_file_keeps_its_entries() {
        let dir = tempfile::tempdir().unwrap();
        let location = active_location(dir.path());

        let storage = TursoStorage::open("kept".to_string()).unwrap();
        assert_eq!(
            storage.insert(b"key", Bytes::from_bytes_vec(vec![7]), Origin::Local),
            Insertion::Stored
        );

        // The registry hands the same database back; go around it to make
        // `migrate` run again on the file as it is on disk.
        let database = block_on(open_database(&location)).unwrap();
        migrate(&database).unwrap();

        let reopened = TursoStorage::open("kept".to_string()).unwrap();
        assert_eq!(reopened.get(b"key"), Some(Bytes::from_bytes_vec(vec![7])));
    }

    /// Two independent connections to one file, which is what two processes
    /// sharing a cache root come down to. Exactly one insert may win, and the
    /// loser must be told which value is actually stored.
    #[test_log::test]
    #[serial_test::serial]
    #[cfg_attr(miri, ignore)]
    fn concurrent_connections_agree_on_the_winner() {
        let dir = tempfile::tempdir().unwrap();
        active_location(dir.path());

        let first = TursoStorage::open("namespace".to_string()).unwrap();
        let second = TursoStorage::open("namespace".to_string()).unwrap();

        let bytes = |value: &[u8]| Bytes::from_bytes_vec(value.to_vec());
        assert_eq!(
            first.insert(b"key", bytes(b"first"), Origin::Local),
            Insertion::Stored
        );

        // The second connection sees the committed entry and leaves it alone.
        assert_eq!(
            second.insert(b"key", bytes(b"second"), Origin::Local),
            Insertion::Conflict(bytes(b"first"))
        );
        assert_eq!(second.get(b"key"), Some(bytes(b"first")));
    }
}
