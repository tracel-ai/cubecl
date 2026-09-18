use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;

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

type DatabaseResult = Result<Arc<turso::Database>, String>;

enum DatabaseState {
    Opening(Vec<async_channel::Sender<DatabaseResult>>),
    Ready(Arc<turso::Database>),
}

static DATABASES: LazyLock<Mutex<HashMap<String, DatabaseState>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub struct TursoStorage {
    database: Arc<turso::Database>,
    namespace: String,
    location: String,
}

impl core::fmt::Debug for TursoStorage {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("TursoStorage")
            .field("namespace", &self.namespace)
            .field("location", &self.location)
            .finish()
    }
}

impl TursoStorage {
    pub async fn open(namespace: String) -> Result<Self, String> {
        let location = location()?;
        let database = shared_database(&location).await?;

        Ok(Self {
            database,
            namespace,
            location,
        })
    }

    async fn connection(&self) -> Result<turso::Connection, String> {
        connect(&self.database).await.map_err(error)
    }

    /// One insert on `connection`: the statement arbitrates, and only a
    /// declined write costs a second one, to fetch the value that won.
    async fn insert_on(
        &self,
        connection: &turso::Connection,
        key: &[u8],
        value: &[u8],
        origin: Origin,
    ) -> Result<Insertion, turso::Error> {
        let changed = connection
            .execute(
                INSERT,
                (
                    self.namespace.as_str(),
                    key.to_vec(),
                    value.to_vec(),
                    origin_code(origin),
                ),
            )
            .await?;
        if changed == 1 {
            return Ok(Insertion::Stored);
        }

        let mut rows = connection
            .query(SELECT, (self.namespace.as_str(), key.to_vec()))
            .await?;
        match rows.next().await? {
            Some(row) => Ok(Insertion::Conflict(Bytes::from_bytes_vec(row.get(0)?))),
            // Purged between the two statements: nothing to report as the
            // winner, and nothing stored.
            None => Ok(Insertion::Failed(
                "the entry was removed while it was being written".to_string(),
            )),
        }
    }

    pub async fn summary() -> Vec<NamespaceSummary> {
        let Ok(location) = location() else {
            return Vec::new();
        };
        let Ok(database) = shared_database(&location).await else {
            return Vec::new();
        };
        let Ok(connection) = connect(&database).await else {
            return Vec::new();
        };
        let Ok(mut rows) = connection
            .query(
                "SELECT namespace, COUNT(*), SUM(length(key) + length(value)) \
                 FROM entries GROUP BY namespace ORDER BY namespace",
                (),
            )
            .await
        else {
            return Vec::new();
        };

        let mut summaries = Vec::new();
        while let Ok(Some(row)) = rows.next().await {
            let Ok(namespace) = row.get::<String>(0) else {
                continue;
            };
            summaries.push(NamespaceSummary {
                namespace,
                entries: row.get::<i64>(1).unwrap_or_default() as u64,
                bytes: row.get::<i64>(2).unwrap_or_default() as u64,
            });
        }
        summaries
    }
}

#[async_trait::async_trait]
impl Storage for TursoStorage {
    async fn get(&self, key: &[u8]) -> Option<Bytes> {
        let result: Result<Option<Bytes>, String> = async {
            let connection = self.connection().await?;
            let mut rows = connection
                .query(SELECT, (self.namespace.as_str(), key.to_vec()))
                .await
                .map_err(error)?;
            let Some(row) = rows.next().await.map_err(error)? else {
                return Ok(None);
            };
            let value: Vec<u8> = row.get(0).map_err(error)?;
            Ok(Some(Bytes::from_bytes_vec(value)))
        }
        .await;

        match result {
            Ok(value) => value,
            Err(error) => {
                log::warn!("Unable to read {}: {error}", self.describe());
                None
            }
        }
    }

    async fn insert(&self, key: &[u8], value: Bytes, origin: Origin) -> Insertion {
        let result = async {
            let connection = self.connection().await?;
            self.insert_on(&connection, key, &value, origin)
                .await
                .map_err(error)
        }
        .await;

        result.unwrap_or_else(Insertion::Failed)
    }

    async fn replace(&self, key: &[u8], value: Bytes, origin: Origin) -> Insertion {
        let result = async {
            let connection = self.connection().await?;
            connection
                .execute(
                    REPLACE,
                    (
                        self.namespace.as_str(),
                        key.to_vec(),
                        value.to_vec(),
                        origin_code(origin),
                    ),
                )
                .await
                .map_err(error)?;
            Ok::<_, String>(Insertion::Stored)
        }
        .await;

        result.unwrap_or_else(Insertion::Failed)
    }

    async fn insert_many(
        &self,
        entries: &mut (dyn Iterator<Item = (Bytes, Bytes)> + Send),
        origin: Origin,
    ) -> InsertSummary {
        // One transaction for the batch: it holds the writer once rather
        // than once per entry, and lands as a whole.
        let refused = |entries: &mut dyn Iterator<Item = (Bytes, Bytes)>| InsertSummary {
            failed: entries.count(),
            ..InsertSummary::default()
        };
        let Ok(mut connection) = connect(&self.database).await else {
            return refused(entries);
        };
        let Ok(transaction) = connection
            .transaction_with_behavior(write_transaction())
            .await
        else {
            return refused(entries);
        };

        let mut summary = InsertSummary::default();
        for (key, value) in entries {
            match self.insert_on(&transaction, &key, &value, origin).await {
                Ok(insertion) => summary.record(&insertion),
                Err(_) => summary.failed += 1,
            }
        }
        if transaction.commit().await.is_err() {
            summary.failed += summary.stored;
            summary.stored = 0;
        }
        summary
    }

    async fn scan(&self) -> Vec<(Bytes, Bytes)> {
        let result = async {
            let connection = self.connection().await?;
            let mut rows = connection
                .query(
                    "SELECT key, value FROM entries WHERE namespace = ?1",
                    (self.namespace.as_str(),),
                )
                .await
                .map_err(error)?;
            let mut entries = Vec::new();
            while let Some(row) = rows.next().await.map_err(error)? {
                let key: Vec<u8> = row.get(0).map_err(error)?;
                let value: Vec<u8> = row.get(1).map_err(error)?;
                entries.push((Bytes::from_bytes_vec(key), Bytes::from_bytes_vec(value)));
            }
            Ok::<_, String>(entries)
        }
        .await;

        result.unwrap_or_else(|error| {
            log::warn!("Unable to scan {}: {error}", self.describe());
            Vec::new()
        })
    }

    async fn purge(&self) {
        if let Ok(connection) = self.connection().await
            && let Err(error) = connection
                .execute(
                    "DELETE FROM entries WHERE namespace = ?1",
                    (self.namespace.as_str(),),
                )
                .await
        {
            log::warn!("Unable to purge {}: {error}", self.describe());
        }
    }

    async fn purge_key(&self, key: &[u8]) {
        if let Ok(connection) = self.connection().await
            && let Err(error) = connection
                .execute(
                    "DELETE FROM entries WHERE namespace = ?1 AND key = ?2",
                    (self.namespace.as_str(), key.to_vec()),
                )
                .await
        {
            log::warn!("Unable to purge a key from {}: {error}", self.describe());
        }
    }

    fn describe(&self) -> String {
        format!("Turso {} ({})", self.location, self.namespace)
    }
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

    let writable = async {
        let database = open_database(location).await?;
        migrate(&database).await?;
        Ok::<_, turso::Error>(database)
    }
    .await;

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
    let connection = connect(&database).await.map_err(error)?;

    let expected = SCHEMA_VERSION.to_string();
    match meta_get(&connection, SCHEMA_VERSION_KEY)
        .await
        .map_err(error)?
    {
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

/// Opens the active environment's database ahead of any storage on it.
pub(crate) async fn open_database_ahead() -> Result<(), String> {
    shared_database(&location()?).await.map(|_| ())
}

/// Whether the active environment's database is open, so a storage on it
/// opens without waiting on anything.
pub(crate) fn database_open() -> bool {
    location().is_ok_and(|location| {
        matches!(
            DATABASES.lock().get(&location),
            Some(DatabaseState::Ready(_))
        )
    })
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
pub(crate) async fn migrate(database: &turso::Database) -> Result<(), turso::Error> {
    let mut connection = connect(database).await?;
    connection.execute(CREATE_META, ()).await?;

    let transaction = connection
        .transaction_with_behavior(write_transaction())
        .await?;

    let expected = SCHEMA_VERSION.to_string();
    let found = meta_get(&transaction, SCHEMA_VERSION_KEY).await?;

    if found.as_deref() != Some(expected.as_str()) {
        match &found {
            Some(found) => log::warn!(
                "cubecl cache: database schema {found} is not {expected}, discarding cached entries"
            ),
            // No version at all: the file predates the `meta` table, so
            // whatever entries it holds cannot be trusted either.
            None => log::debug!("cubecl cache: initializing database schema {expected}"),
        }
        transaction.execute(DROP_ENTRIES, ()).await?;
        meta_set(&transaction, SCHEMA_VERSION_KEY, &expected).await?;
    }

    transaction.execute(CREATE_ENTRIES, ()).await?;
    transaction.commit().await
}

/// Reads a `meta` row, or `None` when the key is absent.
///
/// This module owns the table, so everything that touches it goes through
/// here: the schema version, and a bundle's manifest.
pub(crate) async fn meta_get(
    connection: &turso::Connection,
    key: &str,
) -> Result<Option<String>, turso::Error> {
    match connection.query(META_GET, (key,)).await?.next().await? {
        Some(row) => Ok(Some(row.get(0)?)),
        None => Ok(None),
    }
}

/// Writes a `meta` row, replacing the key's previous value.
pub(crate) async fn meta_set(
    connection: &turso::Connection,
    key: &str,
    value: &str,
) -> Result<(), turso::Error> {
    connection.execute(META_SET, (key, value)).await?;
    Ok(())
}

/// Folds the WAL into the main file and truncates it, so the file stands on
/// its own. Turso never checkpoints on close: a database file copied without
/// its `-wal` is a database missing every write since the last checkpoint.
///
/// Reports whether the checkpoint completed; it doesn't when another
/// connection holds the WAL, and Turso folds every other cause into the same
/// flag while logging the reason itself.
#[cfg(native_cache)]
pub(crate) async fn checkpoint(connection: &turso::Connection) -> Result<bool, turso::Error> {
    // The closure's error type is the SDK's, not this crate's; a row that
    // doesn't decode is read as "did not complete" rather than converted.
    let mut incomplete = false;
    connection
        .pragma_query("wal_checkpoint(TRUNCATE)", |row| {
            incomplete |= row.get::<i64>(0).map_or(true, |busy| busy != 0);
            Ok(())
        })
        .await?;
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
pub(crate) async fn connect(database: &turso::Database) -> Result<turso::Connection, turso::Error> {
    let connection = database.connect()?;
    connection.busy_timeout(BUSY_TIMEOUT)?;
    // A `PRAGMA` that assigns answers with no rows, which `execute` reports as
    // `Misuse`; `pragma_query` takes it either way.
    connection
        .pragma_query("synchronous = NORMAL", |_| Ok(()))
        .await?;
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

pub async fn open(namespace: &str) -> Result<Arc<dyn Storage>, String> {
    TursoStorage::open(namespace.to_string())
        .await
        .map(|storage| Arc::new(storage) as Arc<dyn Storage>)
}

#[cfg(all(test, native_cache))]
mod tests {
    use super::*;
    use alloc::vec;

    /// The tables a database file holds, by name.
    async fn tables(location: &str) -> Vec<String> {
        let database = open_database(location).await.unwrap();
        let connection = connect(&database).await.unwrap();
        let mut rows = connection
            .query(
                "SELECT name FROM sqlite_schema WHERE type = 'table' ORDER BY name",
                (),
            )
            .await
            .unwrap();

        let mut names = Vec::new();
        while let Some(row) = rows.next().await.unwrap() {
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
    #[tokio::test]
    #[serial_test::serial]
    #[cfg_attr(miri, ignore)]
    async fn an_incompatible_schema_is_rebuilt() {
        let dir = tempfile::tempdir().unwrap();
        let location = active_location(dir.path());

        {
            let database = open_database(&location).await.unwrap();
            let connection = connect(&database).await.unwrap();
            connection.execute(CREATE_META, ()).await.unwrap();
            connection
                .execute(META_SET, (SCHEMA_VERSION_KEY, "999"))
                .await
                .unwrap();
            connection
                .execute(
                    "CREATE TABLE entries (store TEXT NOT NULL, key BLOB NOT NULL, \
                     value BLOB NOT NULL, PRIMARY KEY (store, key))",
                    (),
                )
                .await
                .unwrap();
            connection
                .execute("INSERT INTO entries VALUES ('old', X'01', X'02')", ())
                .await
                .unwrap();
        }

        let storage = TursoStorage::open("old".to_string()).await.unwrap();
        assert_eq!(storage.get(b"\x01").await, None, "stale rows are gone");
        // The rebuilt table must be usable, which an emptied one would not be.
        assert_eq!(
            storage
                .insert(b"key", Bytes::from_bytes_vec(vec![1]), Origin::Local)
                .await,
            Insertion::Stored,
            "the rebuilt table accepts the current column layout"
        );

        assert_eq!(tables(&location).await, vec!["entries", "meta"]);

        let database = open_database(&location).await.unwrap();
        let connection = connect(&database).await.unwrap();
        let mut rows = connection
            .query(META_GET, (SCHEMA_VERSION_KEY,))
            .await
            .unwrap();
        let version: String = rows.next().await.unwrap().unwrap().get(0).unwrap();
        assert_eq!(version, SCHEMA_VERSION.to_string());
    }

    /// An opener dropped before it finished — a timeout, a panic — must
    /// settle the registry: the location is free to open again, and whoever
    /// was waiting on it is told rather than left waiting forever.
    #[tokio::test]
    #[serial_test::serial]
    async fn a_cancelled_open_releases_the_location() {
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
        assert!(receiver.recv().await.unwrap().is_err());
    }

    /// The version is written once and survives reopening; entries do too,
    /// because a file already at this version is not rebuilt.
    #[tokio::test]
    #[serial_test::serial]
    #[cfg_attr(miri, ignore)]
    async fn a_current_file_keeps_its_entries() {
        let dir = tempfile::tempdir().unwrap();
        let location = active_location(dir.path());

        let storage = TursoStorage::open("kept".to_string()).await.unwrap();
        assert_eq!(
            storage
                .insert(b"key", Bytes::from_bytes_vec(vec![7]), Origin::Local)
                .await,
            Insertion::Stored
        );

        // The registry hands the same database back; go around it to make
        // `migrate` run again on the file as it is on disk.
        let database = open_database(&location).await.unwrap();
        migrate(&database).await.unwrap();

        let reopened = TursoStorage::open("kept".to_string()).await.unwrap();
        assert_eq!(
            reopened.get(b"key").await,
            Some(Bytes::from_bytes_vec(vec![7]))
        );
    }
}
