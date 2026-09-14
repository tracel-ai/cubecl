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

const CREATE_SCHEMA: &str = "
    CREATE TABLE IF NOT EXISTS cache_entries (
        namespace TEXT NOT NULL,
        key BLOB NOT NULL,
        value BLOB NOT NULL,
        origin INTEGER NOT NULL,
        PRIMARY KEY (namespace, key)
    )
";

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
        self.database.connect().map_err(error)
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
        let mut connection = self.database.connect()?;
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
        let Ok(connection) = database.connect() else {
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
        let Ok(mut connection) = self.database.connect() else {
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
        let database = open_database(location).await?;
        let connection = database.connect().map_err(error)?;
        connection.execute(CREATE_SCHEMA, ()).await.map_err(error)?;
        Ok(Arc::new(database))
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
