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
///
/// Versions 1 to 3 were written by the rusqlite backend into a table named
/// `entries`; opening such a file drops that table too.
pub const SCHEMA_VERSION: u32 = 4;

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
    CREATE TABLE IF NOT EXISTS cache_entries (
        namespace TEXT NOT NULL,
        key BLOB NOT NULL,
        value BLOB NOT NULL,
        origin INTEGER NOT NULL,
        PRIMARY KEY (namespace, key)
    )
";

/// The tables a schema change leaves behind: this build's, and the rusqlite
/// backend's.
const DROP_ENTRIES: [&str; 2] = [
    "DROP TABLE IF EXISTS cache_entries",
    "DROP TABLE IF EXISTS entries",
];

type DatabaseResult = Result<Arc<turso::Database>, String>;

enum DatabaseState {
    Opening(Vec<async_channel::Sender<DatabaseResult>>),
    Ready(Arc<turso::Database>),
}

static DATABASES: LazyLock<Mutex<HashMap<String, DatabaseState>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

#[derive(Clone)]
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

    fn connection(&self) -> Result<turso::Connection, String> {
        connect(&self.database).map_err(error)
    }

    async fn insert_on(
        &self,
        connection: &turso::Connection,
        key: &[u8],
        value: &[u8],
        origin: Origin,
    ) -> Result<Insertion, turso::Error> {
        let mut rows = connection
            .query(
                "SELECT value, origin FROM cache_entries WHERE namespace = ?1 AND key = ?2",
                (self.namespace.as_str(), key.to_vec()),
            )
            .await?;

        if let Some(row) = rows.next().await? {
            let existing: Vec<u8> = row.get(0)?;
            let existing_origin: i64 = row.get(1)?;
            if !(origin == Origin::Local && existing_origin == origin_code(Origin::Imported)) {
                return Ok(Insertion::Conflict(Bytes::from_bytes_vec(existing)));
            }

            connection
                .execute(
                    "UPDATE cache_entries SET value = ?3, origin = ?4 \
                     WHERE namespace = ?1 AND key = ?2",
                    (
                        self.namespace.as_str(),
                        key.to_vec(),
                        value.to_vec(),
                        origin_code(origin),
                    ),
                )
                .await?;
            return Ok(Insertion::Stored);
        }

        connection
            .execute(
                "INSERT INTO cache_entries (namespace, key, value, origin) VALUES (?1, ?2, ?3, ?4)",
                (
                    self.namespace.as_str(),
                    key.to_vec(),
                    value.to_vec(),
                    origin_code(origin),
                ),
            )
            .await?;
        Ok(Insertion::Stored)
    }

    async fn insert_transactional(
        &self,
        key: &[u8],
        value: &[u8],
        origin: Origin,
    ) -> Result<Insertion, turso::Error> {
        let mut connection = connect(&self.database)?;
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .await?;
        let insertion = self.insert_on(&transaction, key, value, origin).await?;
        transaction.commit().await?;
        Ok(insertion)
    }

    pub async fn summary() -> Vec<NamespaceSummary> {
        let Ok(location) = location() else {
            return Vec::new();
        };
        let Ok(database) = shared_database(&location).await else {
            return Vec::new();
        };
        let Ok(connection) = connect(&database) else {
            return Vec::new();
        };
        let Ok(mut rows) = connection
            .query(
                "SELECT namespace, COUNT(*), SUM(length(key) + length(value)) \
                 FROM cache_entries GROUP BY namespace ORDER BY namespace",
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
            let connection = self.connection()?;
            let mut rows = connection
                .query(
                    "SELECT value FROM cache_entries WHERE namespace = ?1 AND key = ?2",
                    (self.namespace.as_str(), key.to_vec()),
                )
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
        self.insert_transactional(key, &value, origin)
            .await
            .unwrap_or_else(|error| Insertion::Failed(error.to_string()))
    }

    async fn replace(&self, key: &[u8], value: Bytes, origin: Origin) -> Insertion {
        let result = async {
            let connection = self.connection()?;
            connection
                .execute(
                    "INSERT INTO cache_entries (namespace, key, value, origin) VALUES (?1, ?2, ?3, ?4) \
                     ON CONFLICT(namespace, key) DO UPDATE SET value = excluded.value, origin = excluded.origin",
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
        let Ok(mut connection) = connect(&self.database) else {
            return InsertSummary {
                failed: entries.count(),
                ..InsertSummary::default()
            };
        };
        let Ok(transaction) = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .await
        else {
            return InsertSummary {
                failed: entries.count(),
                ..InsertSummary::default()
            };
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
            let connection = self.connection()?;
            let mut rows = connection
                .query(
                    "SELECT key, value FROM cache_entries WHERE namespace = ?1",
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
        if let Ok(connection) = self.connection()
            && let Err(error) = connection
                .execute(
                    "DELETE FROM cache_entries WHERE namespace = ?1",
                    (self.namespace.as_str(),),
                )
                .await
        {
            log::warn!("Unable to purge {}: {error}", self.describe());
        }
    }

    async fn purge_key(&self, key: &[u8]) {
        if let Ok(connection) = self.connection()
            && let Err(error) = connection
                .execute(
                    "DELETE FROM cache_entries WHERE namespace = ?1 AND key = ?2",
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

    let opened = async {
        let writable = async {
            let database = open_database(location).await?;
            migrate(&database).await.map_err(error)?;
            Ok::<_, String>(database)
        }
        .await;

        match writable {
            Ok(database) => Ok(Arc::new(database)),
            #[cfg(native_cache)]
            Err(err) => open_read_only(location, &err).await.map(Arc::new),
            #[cfg(not(native_cache))]
            Err(err) => Err(err),
        }
    }
    .await;
    let waiters = {
        let mut databases = DATABASES.lock();
        let waiters = match databases.remove(location) {
            Some(DatabaseState::Opening(waiters)) => waiters,
            _ => Vec::new(),
        };
        if let Ok(database) = &opened {
            databases.insert(location.to_string(), DatabaseState::Ready(database.clone()));
        }
        waiters
    };

    for waiter in waiters {
        let _ = waiter.try_send(opened.clone());
    }
    opened
}

#[cfg(native_cache)]
async fn open_database(location: &str) -> Result<turso::Database, String> {
    if let Some(parent) = std::path::Path::new(location).parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("unable to create cache directory: {error}"))?;
    }

    turso::Builder::new_local(location)
        .experimental_multiprocess_wal(true)
        .build()
        .await
        .map_err(error)
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
async fn open_database(location: &str) -> Result<turso::Database, String> {
    let wal = format!("{location}-wal");
    let io = super::turso_browser::BrowserIo::new(&[location, &wal]).await?;
    turso::Builder::new_local(location)
        .with_io_impl(Arc::new(io))
        .build()
        .await
        .map_err(error)
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

/// Brings the file to [`SCHEMA_VERSION`], dropping the entries of any other.
///
/// The table is dropped rather than emptied: a schema change can rename or
/// retype a column, and keeping the old table would make every later statement
/// fail instead of costing one cold start. The version is read and the schema
/// rebuilt under one write lock, so two processes opening the same file at
/// once rebuild it once: the second waits, then reads the version the first
/// wrote.
pub(crate) async fn migrate(database: &turso::Database) -> Result<(), turso::Error> {
    let mut connection = connect(database)?;
    connection.execute(CREATE_META, ()).await?;

    let transaction = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
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
        for drop in DROP_ENTRIES {
            transaction.execute(drop, ()).await?;
        }
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

/// A connection to `database` with the busy timeout set.
pub(crate) fn connect(database: &turso::Database) -> Result<turso::Connection, turso::Error> {
    let connection = database.connect()?;
    connection.busy_timeout(BUSY_TIMEOUT)?;
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
        let connection = connect(&database).unwrap();
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

    /// A file written by the rusqlite backend carries its `entries` table and
    /// an older schema version. Opening it must leave neither behind: the
    /// entries are unreadable to this build, and the stale table would
    /// otherwise sit in the file forever.
    #[tokio::test]
    #[serial_test::serial]
    #[cfg_attr(miri, ignore)]
    async fn a_legacy_file_is_rebuilt_on_open() {
        let dir = tempfile::tempdir().unwrap();
        let location = active_location(dir.path());

        {
            let database = open_database(&location).await.unwrap();
            let connection = connect(&database).unwrap();
            connection.execute(CREATE_META, ()).await.unwrap();
            connection
                .execute(META_SET, (SCHEMA_VERSION_KEY, "3"))
                .await
                .unwrap();
            connection
                .execute(
                    "CREATE TABLE entries (namespace TEXT, key BLOB, value BLOB, origin INTEGER)",
                    (),
                )
                .await
                .unwrap();
            connection
                .execute("INSERT INTO entries VALUES ('old', X'01', X'02', 0)", ())
                .await
                .unwrap();
        }

        let storage = TursoStorage::open("migrated".to_string()).await.unwrap();
        assert_eq!(storage.get(b"\x01").await, None);

        assert_eq!(tables(&location).await, vec!["cache_entries", "meta"]);

        let database = open_database(&location).await.unwrap();
        let connection = connect(&database).unwrap();
        let mut rows = connection
            .query(META_GET, (SCHEMA_VERSION_KEY,))
            .await
            .unwrap();
        let version: String = rows.next().await.unwrap().unwrap().get(0).unwrap();
        assert_eq!(version, SCHEMA_VERSION.to_string());
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
