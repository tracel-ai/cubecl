//! The database file shared by every namespace of an environment, on the
//! engine [`sql`] names.
//!
//! Every statement runs to completion on the spot, so the [`Storage`] this
//! module hands out is synchronous like every other. Opening is the
//! exception: the browser reaches its files through promises. The database
//! therefore opens through an `async` step ([`open_ahead`]), which a page
//! awaits once before anything needs it; natively a storage opens the
//! database itself on first use, blocking on the file system as a file read
//! would.

use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ops::Range;

use hashbrown::HashMap;

use super::sql::{self, Behavior, Opening, SqlError, Transaction};
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

/// The columns of the entries table, which [`CREATE_ENTRIES`] and
/// [`CREATE_INCOMING`] share.
macro_rules! entries_columns {
    () => {
        "(
            namespace TEXT NOT NULL,
            key BLOB NOT NULL,
            value BLOB NOT NULL,
            origin INTEGER NOT NULL,
            PRIMARY KEY (namespace, key)
        )"
    };
}

const CREATE_ENTRIES: &str = concat!("CREATE TABLE IF NOT EXISTS entries ", entries_columns!());

/// The table a replacement fills before it takes the place of the entries
/// ([`copy_in_batches`]).
#[cfg(native_cache)]
const CREATE_INCOMING: &str = concat!("CREATE TABLE entries_incoming ", entries_columns!());

#[cfg(native_cache)]
const DROP_INCOMING: &str = "DROP TABLE IF EXISTS entries_incoming";

#[cfg(native_cache)]
const INSERT_INCOMING: &str = "INSERT INTO entries_incoming (namespace, key, value, origin) \
                               VALUES (?1, ?2, ?3, ?4)";

/// Run in one write, so that the replacement lands as a whole.
#[cfg(native_cache)]
const SWAP_INCOMING: [&str; 2] = [
    "DROP TABLE entries",
    "ALTER TABLE entries_incoming RENAME TO entries",
];

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

const SCAN: &str = "SELECT key, value, origin FROM entries WHERE namespace = ?1";

#[cfg(native_cache)]
const ENTRIES: &str = "SELECT namespace, key, value FROM entries";

const PURGE: &str = "DELETE FROM entries WHERE namespace = ?1";

#[cfg(native_cache)]
const CLEAR: &str = "DELETE FROM entries";

const PURGE_KEY: &str = "DELETE FROM entries WHERE namespace = ?1 AND key = ?2";

const PURGE_RANGE: &str = "DELETE FROM entries WHERE namespace >= ?1 AND namespace < ?2";

const SUMMARY: &str = "SELECT namespace, COUNT(*), SUM(length(key) + length(value)) \
                       FROM entries GROUP BY namespace ORDER BY namespace";

const NAMESPACES: &str = "SELECT DISTINCT namespace FROM entries ORDER BY namespace";

type DatabaseResult = Result<Arc<sql::Database>, String>;

enum DatabaseState {
    Opening(Vec<async_channel::Sender<DatabaseResult>>),
    Ready(Arc<sql::Database>),
}

/// The databases this process has opened, by location, so that every
/// namespace of an environment shares one.
static DATABASES: LazyLock<Mutex<HashMap<String, DatabaseState>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// What [`Database::scan_with_origin`] calls per entry: its key, its value, and
/// where it came from.
type OriginVisitor<'a> = dyn FnMut(&[u8], &[u8], Origin) + 'a;

/// How a database opened at a path ([`Database::open_at`]) may be used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    /// Read and never written: not migrated, not checkpointed, and refused
    /// unless the file is an environment at this build's schema version.
    /// Safe to point at a file another process is writing. Standalone files
    /// need no writable directory; a live shared Turso WAL requires write
    /// access to its existing coordination sidecar even for readers.
    ///
    /// The engine keeps one database object per file for the whole process,
    /// configured by the first open: while a read-only handle lives, every
    /// other open of that file in the process is read-only too, including the
    /// stores of an environment activated on it meanwhile, for as long as they
    /// live. The active environment's own file is the exception: a read-only
    /// handle to it shares the active environment's database rather than
    /// downgrading it.
    ReadOnly,
    /// Read and written, and brought to this build's schema on open exactly
    /// as the active environment's database is: a file of another schema has
    /// its entries dropped.
    ReadWrite,
}

/// Every namespace of an environment's database file at once: the active
/// environment's ([`open_active`](Self::open_active)), or any other
/// ([`open_at`](Self::open_at)).
///
/// A [`DatabaseStorage`] is this bound to one namespace. The records of a build
/// ([`crate::records`]) span namespaces — the sessions', and one per record
/// kind — and are read and written through this instead.
pub struct Database {
    /// The engine rejects concurrent use of a connection at runtime rather
    /// than serializing it, so each handle holds one and takes it in turn.
    connection: Mutex<sql::Connection>,
    location: String,
    /// Read by [`into_standalone`](Self::into_standalone), which only a file
    /// system has.
    #[cfg_attr(browser_cache, expect(dead_code, reason = "native only"))]
    access: Access,
    /// The bytes one frame of the file's log takes: a page and its header.
    #[cfg_attr(browser_cache, expect(dead_code, reason = "native only"))]
    frame_bytes: u64,
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
        let frame_bytes = frame_bytes(&connection).map_err(error)?;

        Ok(Self {
            connection: Mutex::new(connection),
            location,
            access: Access::ReadWrite,
            frame_bytes,
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
                log::warn!("Unable to open the {} cache: {error}", sql::ENGINE);
                None
            }
        }
    }

    /// The environment database file at `path`, whichever environment is
    /// active: a saved environment, one a build ran in, another process's.
    ///
    /// The engine shares one database object per file across the process, so
    /// opening a file a store of the active environment already has open is
    /// the same database, not a second writer beside it. The active
    /// environment's own file is opened as its stores open it, whatever
    /// `access` asks: a read-only open there would leave its stores unable to
    /// write.
    ///
    /// # Errors
    ///
    /// When `path` is not UTF-8, when the file can't be opened with `access`,
    /// and, read-only, when it is missing or is not an environment at this
    /// build's schema.
    #[cfg(native_cache)]
    pub fn open_at(path: &std::path::Path, access: Access) -> Result<Self, String> {
        let active = location()
            .ok()
            .filter(|active| same_file(path, std::path::Path::new(active)));
        let (location, database) = if let Some(active) = active {
            let database = crate::future::block_on(shared_database(&active))?;
            (active, database)
        } else {
            let location = path
                .to_str()
                .ok_or_else(|| format!("database path {path:?} is not valid UTF-8"))?
                .to_string();
            let database = crate::future::block_on(open_at_path(&location, access))?;
            (location, Arc::new(database))
        };
        let connection = connect(&database).map_err(error)?;
        let frame_bytes = frame_bytes(&connection).map_err(error)?;

        Ok(Self {
            connection: Mutex::new(connection),
            location,
            access,
            frame_bytes,
        })
    }

    /// Runs `operation` on the connection, for the table beside the entries
    /// (a bundle's manifest).
    #[cfg(native_cache)]
    pub(crate) fn with_connection<T>(&self, operation: impl FnOnce(&sql::Connection) -> T) -> T {
        operation(&self.connection.lock())
    }

    /// Runs `operation` on the connection, logging a failure under `name`
    /// against `namespace` and reporting it as a message.
    fn run<T>(
        &self,
        name: &str,
        namespace: &str,
        operation: impl FnOnce(&sql::Connection) -> Result<T, SqlError>,
    ) -> Result<T, String> {
        let connection = self.connection.lock();
        caught(|| operation(&connection))
            .map_err(|fault| self.failed(name, namespace, fault.explain(&self.location)))
    }

    /// [`run`](Self::run) for a write: the log is first given room for it
    /// ([`make_log_room`]), and a log that can't be given room refuses it.
    fn write<T>(
        &self,
        name: &str,
        namespace: &str,
        operation: impl FnOnce(&sql::Connection) -> Result<T, SqlError>,
    ) -> Result<T, String> {
        let connection = self.connection.lock();
        self.make_log_room(&connection)
            .map_err(|reason| self.failed(name, namespace, reason))?;
        caught(|| operation(&connection))
            .map_err(|fault| self.failed(name, namespace, fault.explain(&self.location)))
    }

    /// The failure to `name` `namespace` for `reason`, which names the file, the cause and
    /// what to do: logged, and reported.
    fn failed(&self, name: &str, namespace: &str, reason: String) -> String {
        let failure = format!(
            "unable to {name} {}: {reason}",
            describe(&self.location, namespace)
        );
        log::warn!("{failure}");
        failure
    }

    /// [`make_log_room`] for this database's log.
    fn make_log_room(&self, connection: &sql::Connection) -> Result<(), String> {
        #[cfg(native_cache)]
        {
            make_log_room(connection, &self.location, self.frame_bytes)
        }
        #[cfg(not(native_cache))]
        {
            let _ = connection;
            Ok(())
        }
    }

    /// The value stored under `key` in `namespace`.
    pub fn get(&self, namespace: &str, key: &[u8]) -> Option<Bytes> {
        self.run("read", namespace, |connection| {
            select_on(connection, namespace, key)
        })
        .unwrap_or_default()
    }

    /// Stores `value` under `key`. See [`Storage`] for the rules; in short, a
    /// local value replaces an imported one and nothing else overwrites.
    pub fn insert(&self, namespace: &str, key: &[u8], value: &[u8], origin: Origin) -> Insertion {
        self.write("write", namespace, |connection| {
            insert_on(connection, namespace, key, value, origin)
        })
        .unwrap_or_else(Insertion::Failed)
    }

    /// Stores `value` under `key`, overwriting whatever is there.
    pub fn replace(&self, namespace: &str, key: &[u8], value: &[u8], origin: Origin) -> Insertion {
        self.write("replace", namespace, |connection| {
            let params = [
                namespace.into(),
                key.into(),
                value.into(),
                origin_code(origin).into(),
            ];
            connection.execute(REPLACE, &params)?;
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
        let connection = self.connection.lock();
        let began = self
            .make_log_room(&connection)
            .and_then(|()| Transaction::new(&connection, write_transaction()).map_err(error));
        let transaction = match began {
            Ok(transaction) => transaction,
            Err(err) => {
                self.failed("batch write", namespace, err);
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
        if let Err(err) = transaction.commit() {
            log::warn!(
                "Unable to commit a batch write to {}: {err}",
                describe(&self.location, namespace)
            );
            summary.failed += summary.stored;
            summary.stored = 0;
        }
        summary
    }

    /// Replaces every entry with those of `source`, stored as
    /// [`Origin::Imported`], or with nothing without one, as a whole: a
    /// failure leaves the entries as they were.
    ///
    /// The entries of `source` are copied into a table beside the entries in
    /// batches of at most [`COPY_BATCH_BYTES`], each its own write, and that
    /// table then takes the entries' place in one small write. A whole
    /// environment copied in one write would write about two log frames a
    /// page, which for one of 512 MiB is the 262,144 frames turso indexes
    /// between processes, past which every read of the file fails. The file
    /// holds both until the swap, and keeps that size after it: the pages the
    /// old entries held are free for later writes, the next replacement's
    /// among them, but not given back.
    ///
    /// # Errors
    ///
    /// When `source` can't be read or this database can't be written.
    #[cfg(native_cache)]
    pub fn replace_with(&self, source: Option<&Database>) -> Result<(), String> {
        let connection = self.connection.lock();
        let source = source.map(|source| source.connection.lock());
        let room = |connection: &sql::Connection| {
            make_log_room(connection, &self.location, self.frame_bytes)
        };
        copy_in_batches(&connection, source.as_deref(), &room)
            .inspect_err(|err| log::warn!("Unable to replace {}: {err}", self.location))?;
        restart_log(&connection, &self.location);
        Ok(())
    }

    /// Visits every entry of `namespace`.
    pub fn scan(&self, namespace: &str, visit: &mut dyn FnMut(&[u8], &[u8])) {
        self.scan_with_origin(namespace, &mut |key, value, _| visit(key, value));
    }

    /// Deletes every entry of `namespace`. Logs a failed delete rather than
    /// reporting it; see the [`Storage`] contract.
    pub fn purge(&self, namespace: &str) {
        let _ = self.write("purge", namespace, |connection| {
            connection.execute(PURGE, &[namespace.into()]).map(|_| ())
        });
    }

    /// Deletes the entry of `namespace` under `key`, with the same failure
    /// contract as [`purge`](Self::purge).
    pub fn purge_key(&self, namespace: &str, key: &[u8]) {
        let _ = self.write("purge a key from", namespace, |connection| {
            connection
                .execute(PURGE_KEY, &[namespace.into(), key.into()])
                .map(|_| ())
        });
    }

    /// Deletes every entry of each namespace in `namespaces`, compared as bytes, with the same
    /// failure contract as [`purge`](Self::purge).
    pub fn purge_range(&self, namespaces: Range<&str>) {
        let _ = self.write(
            "purge the namespaces from",
            namespaces.start,
            |connection| {
                connection
                    .execute(
                        PURGE_RANGE,
                        &[namespaces.start.into(), namespaces.end.into()],
                    )
                    .map(|_| ())
            },
        );
    }

    /// Visits every entry of `namespace` with the [`Origin`] it was stored
    /// under: whether this environment computed it or imported it.
    pub fn scan_with_origin(&self, namespace: &str, visit: &mut OriginVisitor<'_>) {
        let _ = self.run("scan", namespace, |connection| {
            connection.query(SCAN, &[namespace.into()], &mut |row| {
                let origin = origin_of(row.integer(2)?);
                visit(&row.blob(0)?, &row.blob(1)?, origin);
                Ok::<_, SqlError>(())
            })
        });
    }

    /// Entry count and total size per namespace, for reporting.
    pub fn summary(&self) -> Vec<NamespaceSummary> {
        self.run("summarize", "", summarize).unwrap_or_default()
    }

    /// Whether the file was last left [standalone](Self::into_standalone): a
    /// rollback-journal header, which the first write turns back into a WAL
    /// one.
    #[cfg(native_cache)]
    pub fn is_standalone(&self) -> bool {
        use std::io::{Read, Seek, SeekFrom};

        let mut header = [0u8; 2];
        std::fs::File::open(&self.location)
            .and_then(|mut file| {
                file.seek(SeekFrom::Start(JOURNAL_MODE_BYTES.start))?;
                file.read_exact(&mut header)
            })
            .is_ok_and(|()| header == LEGACY_JOURNAL)
    }

    /// Folds the WAL into the file and marks it a rollback-journal file, so
    /// that it stands on its own and opens from a directory nobody may write,
    /// as an exported bundle does.
    ///
    /// Consumes the handle, whose connection has to be closed before the
    /// header is rewritten. Nothing else in the process may have the file
    /// open: a connection still open to it writes the WAL header back.
    ///
    /// # Errors
    ///
    /// When the handle was opened [`Access::ReadOnly`], when the file is an
    /// environment's this process has open (the active one's, or one it was
    /// active on earlier), and when the checkpoint or the header write fails.
    /// Other handles from [`open_at`](Self::open_at) on the same file are
    /// the caller's to drop first.
    #[cfg(native_cache)]
    pub fn into_standalone(self) -> Result<(), String> {
        let Self {
            connection,
            location,
            access,
            ..
        } = self;
        if access == Access::ReadOnly {
            return Err(format!("{location} was opened read-only"));
        }
        let shared = DATABASES.lock().keys().any(|opened| {
            same_file(
                std::path::Path::new(opened),
                std::path::Path::new(&location),
            )
        });
        if shared {
            return Err(format!(
                "{location} is open as an environment's database in this process"
            ));
        }
        make_standalone(connection.into_inner(), std::path::Path::new(&location))
    }

    /// The names of every namespace this database holds.
    pub fn namespaces(&self) -> Vec<String> {
        self.run("list the namespaces of", "", |connection| {
            let mut namespaces = Vec::new();
            connection.query(NAMESPACES, &[], &mut |row| {
                namespaces.push(row.text(0)?);
                Ok::<_, SqlError>(())
            })?;
            Ok(namespaces)
        })
        .unwrap_or_default()
    }
}

/// An environment's database reads as a bundle: a bundle file carries the
/// same schema, so what one environment holds imports into another the way a
/// shipped bundle does.
impl crate::bundle::Bundle for Database {
    fn get(&self, namespace: &str, key: &[u8]) -> Option<Bytes> {
        Database::get(self, namespace, key)
    }

    fn scan(&self, namespace: &str, visit: &mut dyn FnMut(&[u8], &[u8])) {
        Database::scan(self, namespace, visit);
    }

    fn namespaces(&self) -> Vec<String> {
        Database::namespaces(self)
    }

    fn describe(&self) -> String {
        self.location.clone()
    }
}

/// The most bytes of entries one write of a replacement copies: about 8,000
/// log frames at two a 4 KiB page, a small part of what turso indexes.
#[cfg(native_cache)]
const COPY_BATCH_BYTES: usize = 16 * 1024 * 1024;

/// Replaces the entries of the database behind `connection` with every
/// entry of `source` as [`Origin::Imported`], or empties them without one,
/// `room` giving the log room before each write.
///
/// The entries are copied into a table of their own, [`COPY_BATCH_BYTES`] at a
/// time, which then takes the entries' place in one write
/// ([`SWAP_INCOMING`]): until that write, the entries are as they were, and
/// a failure before it drops the copy and leaves them so.
#[cfg(native_cache)]
fn copy_in_batches(
    connection: &sql::Connection,
    source: Option<&sql::Connection>,
    room: &dyn Fn(&sql::Connection) -> Result<(), String>,
) -> Result<(), String> {
    room(connection)?;
    let Some(source) = source else {
        connection.execute(CLEAR, &[])?;
        return Ok(());
    };
    // Left by a replacement that died before its swap.
    connection.execute(DROP_INCOMING, &[])?;
    connection.execute(CREATE_INCOMING, &[])?;
    let copied = copy_into_incoming(connection, source, room).and_then(|()| {
        room(connection)?;
        let transaction = Transaction::new(connection, write_transaction())?;
        for statement in SWAP_INCOMING {
            transaction.execute(statement, &[])?;
        }
        Ok(transaction.commit()?)
    });
    if copied.is_err() {
        let _ = connection.execute(DROP_INCOMING, &[]);
    }
    copied
}

/// Copies every entry of `source` into the table a replacement fills, one
/// write per [`COPY_BATCH_BYTES`].
#[cfg(native_cache)]
fn copy_into_incoming(
    connection: &sql::Connection,
    source: &sql::Connection,
    room: &dyn Fn(&sql::Connection) -> Result<(), String>,
) -> Result<(), String> {
    let mut batch: Option<(Transaction<'_>, usize)> = None;
    source.query(ENTRIES, &[], &mut |row| {
        if batch.is_none() {
            room(connection)?;
            batch = Some((Transaction::new(connection, write_transaction())?, 0));
        }
        let Some((transaction, bytes)) = batch.as_mut() else {
            return Ok(());
        };
        let namespace = row.text(0)?;
        let key = row.blob(1)?;
        let value = row.blob(2)?;
        *bytes += namespace.len() + key.len() + value.len();
        let params = [
            (&namespace).into(),
            (&key).into(),
            (&value).into(),
            origin_code(Origin::Imported).into(),
        ];
        transaction.execute(INSERT_INCOMING, &params)?;
        if *bytes >= COPY_BATCH_BYTES
            && let Some((transaction, _)) = batch.take()
        {
            transaction.commit()?;
        }
        Ok::<_, String>(())
    })?;
    // A batch that failed partway is rolled back as it drops.
    match batch {
        Some((transaction, _)) => Ok(transaction.commit()?),
        None => Ok(()),
    }
}

/// One insert on `connection`: the statement arbitrates, and only a declined
/// write costs a second one, to fetch the value that won.
fn insert_on(
    connection: &sql::Connection,
    namespace: &str,
    key: &[u8],
    value: &[u8],
    origin: Origin,
) -> Result<Insertion, SqlError> {
    let params = [
        namespace.into(),
        key.into(),
        value.into(),
        origin_code(origin).into(),
    ];
    if connection.execute(INSERT, &params)? == 1 {
        return Ok(Insertion::Stored);
    }

    match select_on(connection, namespace, key)? {
        Some(value) => Ok(Insertion::Conflict(value)),
        // Purged between the two statements: nothing to report as the
        // winner, and nothing stored.
        None => Ok(Insertion::Failed(
            "the entry was removed while it was being written".to_string(),
        )),
    }
}

/// The value stored under `key` in `namespace`, read on `connection`.
fn select_on(
    connection: &sql::Connection,
    namespace: &str,
    key: &[u8],
) -> Result<Option<Bytes>, SqlError> {
    let mut value = None;
    connection.query(SELECT, &[namespace.into(), key.into()], &mut |row| {
        value = Some(Bytes::from_bytes_vec(row.blob(0)?));
        Ok::<_, SqlError>(())
    })?;
    Ok(value)
}

fn describe(location: &str, namespace: &str) -> String {
    format!("{} {location} ({namespace})", sql::ENGINE)
}

/// One namespace of an environment's database file.
pub struct DatabaseStorage {
    database: Database,
    namespace: String,
}

impl core::fmt::Debug for DatabaseStorage {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("DatabaseStorage")
            .field("namespace", &self.namespace)
            .field("location", &self.database.location)
            .finish()
    }
}

impl DatabaseStorage {
    /// Binds a storage to `namespace` in the active environment's database.
    pub fn open(namespace: String) -> Result<Self, String> {
        Ok(Self {
            database: Database::open()?,
            namespace,
        })
    }

    /// Binds a storage to `namespace` in `database`, whichever file it was
    /// opened on: a [`Store`](super::Store) over a saved environment's
    /// namespace, through [`StoreOptions::storage_with`](super::StoreOptions::storage_with).
    pub fn new(database: Database, namespace: String) -> Self {
        Self {
            database,
            namespace,
        }
    }
}

impl Storage for DatabaseStorage {
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
        cached: false,
    };

    let writable = match open_database(location).await {
        Ok(database) => migrate(&database).map(|()| database),
        Err(err) => Err(err),
    };

    opener.result = match writable {
        Ok(database) => {
            opener.cached = true;
            Ok(Arc::new(database))
        }
        // A lock another process holds is not a read-only location. Report
        // it and cache nothing, so the next open tries the writable path
        // again rather than serving a read-only file for the rest of the
        // process.
        Err(err @ SqlError::Busy(_)) => Err(error(err)),
        // Served but not cached: a file is read-only here also while a
        // read-only handle from `Database::open_at` holds it, and caching
        // would keep it read-only for the rest of the process.
        #[cfg(native_cache)]
        Err(err) => {
            log::debug!("cubecl cache: {location} is not writable ({err}); opening read-only");
            open_read_only(location).await.map(Arc::new)
        }
        #[cfg(not(native_cache))]
        Err(err) => Err(error(err)),
    };
    opener.result.clone()
}

/// The registry's `Opening` entry for one location, settled when the opener
/// is dropped: replaced by `Ready` when the result is `cached`, removed
/// otherwise, and every waiter handed the outcome either way.
struct Opener<'a> {
    location: &'a str,
    result: DatabaseResult,
    cached: bool,
}

impl Drop for Opener<'_> {
    fn drop(&mut self) {
        let waiters = {
            let mut databases = DATABASES.lock();
            let waiters = match databases.remove(self.location) {
                Some(DatabaseState::Opening(waiters)) => waiters,
                _ => Vec::new(),
            };
            if let Ok(database) = &self.result
                && self.cached
            {
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

/// Opens `location` writable, shared with every other process using it.
#[cfg(native_cache)]
async fn open_database(location: &str) -> Result<sql::Database, SqlError> {
    if let Some(parent) = std::path::Path::new(location).parent() {
        std::fs::create_dir_all(parent).map_err(|error| {
            SqlError::Io(format!(
                "creating the cache directory {}: {error}",
                parent.display()
            ))
        })?;
    }
    recover_long_log(location).await;
    sql::Database::open(location, Opening::Shared).await
}

#[cfg(browser_cache)]
async fn open_database(location: &str) -> Result<sql::Database, SqlError> {
    sql::Database::open(location, Opening::Shared).await
}

/// What [`recover_long_log`] found and did.
#[cfg(native_cache)]
#[derive(Debug, PartialEq, Eq)]
enum LogRecovery {
    /// The log is short, or there is none.
    NotNeeded,
    /// Another process has the file open, and its log is that process's to
    /// restart.
    #[cfg(not(all(windows, turso_engine)))]
    HeldElsewhere,
    /// Recovery could not complete safely; every cache file stays in place.
    #[cfg(all(windows, turso_engine))]
    Deferred,
    /// The log was folded into the file and restarted.
    Restarted,
    /// The log could not be folded in: the file and its log were moved
    /// aside, so the database opens empty, and is filled again from the
    /// saved environment.
    #[cfg(not(all(windows, turso_engine)))]
    SetAside,
}

/// Recovers a long log left by a process that is gone: one past
/// [`LOG_RESTART_FRAMES`], which a writer that died before restarting it
/// leaves, or one past the 262,144 frames turso indexes between processes,
/// on which every read fails as `database is locked` until the log restarts.
///
/// Windows uses a coordinated checkpoint: Turso's process-local mode does
/// not exclude shared IOCP owners. The coordination file is retained even
/// after success, and an unsuccessful checkpoint leaves every file intact.
/// A log already beyond the shared index's capacity may need offline recovery.
/// Other platforms retain the process-local recovery path below.
#[cfg(native_cache)]
async fn recover_long_log(location: &str) -> LogRecovery {
    if sql::INDEXED_LOG_FRAMES.is_none() {
        return LogRecovery::NotNeeded;
    }
    let Some(page_size) = page_size_on_disk(location) else {
        return LogRecovery::NotNeeded;
    };
    let frames = log_frames(location, page_size + 24);
    if frames < LOG_RESTART_FRAMES {
        return LogRecovery::NotNeeded;
    }

    recover_log(location, frames).await
}

/// Keep IOCP readers and writers in one coordination domain for the entire
/// checkpoint. Neither completion nor failure proves exclusive file ownership.
#[cfg(all(native_cache, windows, turso_engine))]
async fn recover_log(location: &str, frames: u64) -> LogRecovery {
    let restarted = match sql::Database::open(location, Opening::Shared).await {
        Ok(database) => connect(&database).and_then(|connection| connection.checkpoint()),
        Err(err) => Err(err),
    };
    match restarted {
        Ok(true) => {
            log::info!(
                "cubecl cache: restarted the shared log of {location}, which held {frames} frames"
            );
            LogRecovery::Restarted
        }
        outcome => {
            let reason = match outcome {
                Err(err) => err.to_string(),
                _ => "the checkpoint did not complete".to_string(),
            };
            log::debug!(
                "cubecl cache: leaving the log of {location} ({frames} frames) intact: {reason}"
            );
            LogRecovery::Deferred
        }
    }
}

#[cfg(all(native_cache, not(all(windows, turso_engine))))]
async fn recover_log(location: &str, frames: u64) -> LogRecovery {
    let restarted = match sql::Database::open(location, Opening::ProcessLocal).await {
        Ok(database) => connect(&database).and_then(|connection| connection.checkpoint()),
        Err(err @ SqlError::HeldByAnotherProcess(_)) => {
            log::debug!(
                "cubecl cache: the log of {location} holds {frames} frames, and another process \
                 has the file open ({err}); leaving the log to it"
            );
            return LogRecovery::HeldElsewhere;
        }
        Err(err) => Err(err),
    };
    match restarted {
        Ok(true) => {
            // The process-local open just dropped never used the index; with
            // the log empty, nothing in the index is still needed.
            let _ = std::fs::remove_file(format!("{location}-tshm"));
            log::info!(
                "cubecl cache: restarted the log of {location}, which a process that is gone \
                 left holding {frames} frames"
            );
            LogRecovery::Restarted
        }
        outcome => {
            let reason = match outcome {
                Err(err) => err.to_string(),
                _ => "the checkpoint did not complete".to_string(),
            };
            set_aside(location, frames, &reason);
            LogRecovery::SetAside
        }
    }
}

/// Moves the database at `location` and the files beside it to
/// `<location>.stale-<seconds>`, saying so.
#[cfg(all(native_cache, not(all(windows, turso_engine))))]
fn set_aside(location: &str, frames: u64, reason: &str) {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| since.as_secs());
    let aside = format!("{location}.stale-{stamp}");
    for suffix in ["", "-wal", "-tshm"] {
        let _ = std::fs::rename(format!("{location}{suffix}"), format!("{aside}{suffix}"));
    }
    log::warn!(
        "cubecl cache: the log of {location} held {frames} frames and could not be folded into \
         the file ({reason}); moved it to {aside}, and starting the database afresh"
    );
}

/// The page size the database file at `location` declares in its header, or
/// `None` when there is no such file yet.
#[cfg(native_cache)]
fn page_size_on_disk(location: &str) -> Option<u64> {
    use std::io::{Read, Seek, SeekFrom};

    let mut header = [0u8; 2];
    let mut file = std::fs::File::open(location).ok()?;
    file.seek(SeekFrom::Start(16)).ok()?;
    file.read_exact(&mut header).ok()?;
    // Stored big-endian, 1 standing for 65,536, which does not fit.
    Some(match u16::from_be_bytes(header) {
        1 => 65_536,
        0 => 4096,
        size => size as u64,
    })
}

/// A database nobody may write, served as it is: a cache root in a container
/// image layer, a Nix store path, a mounted bundle.
///
/// Only a file already at this build's schema qualifies. Anything else would
/// need the rebuild [`migrate`] performs, which needs a writable file; the
/// caller falls back to memory instead.
#[cfg(native_cache)]
async fn open_read_only(location: &str) -> Result<sql::Database, String> {
    let database = sql::Database::open(location, Opening::ReadOnly)
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

/// A database file opened with `access`, outside the registry: the file is not
/// the active environment's.
#[cfg(native_cache)]
async fn open_at_path(location: &str, access: Access) -> Result<sql::Database, String> {
    match access {
        Access::ReadOnly => open_read_only(location).await,
        Access::ReadWrite => {
            let database = open_database(location).await.map_err(error)?;
            migrate(&database).map_err(error)?;
            Ok(database)
        }
    }
}

/// Whether `a` and `b` name the same file, compared as written when either
/// does not exist yet.
#[cfg(native_cache)]
pub fn same_file(a: &std::path::Path, b: &std::path::Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
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
fn write_transaction() -> Behavior {
    #[cfg(browser_cache)]
    {
        Behavior::Deferred
    }
    #[cfg(not(browser_cache))]
    {
        Behavior::Immediate
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
pub(crate) fn migrate(database: &sql::Database) -> Result<(), SqlError> {
    let connection = connect(database)?;
    connection.execute(CREATE_META, &[])?;

    let transaction = Transaction::new(&connection, write_transaction())?;

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
        transaction.execute(DROP_ENTRIES, &[])?;
        meta_set(&transaction, SCHEMA_VERSION_KEY, &expected)?;
    }

    transaction.execute(CREATE_ENTRIES, &[])?;
    transaction.commit()
}

/// Reads a `meta` row, or `None` when the key is absent.
///
/// This module owns the table, so everything that touches it goes through
/// here: the schema version, and a bundle's manifest.
pub(crate) fn meta_get(
    connection: &sql::Connection,
    key: &str,
) -> Result<Option<String>, SqlError> {
    let mut value = None;
    connection.query(META_GET, &[key.into()], &mut |row| {
        value = Some(row.text(0)?);
        Ok::<_, SqlError>(())
    })?;
    Ok(value)
}

/// Writes a `meta` row, replacing the key's previous value.
pub(crate) fn meta_set(
    connection: &sql::Connection,
    key: &str,
    value: &str,
) -> Result<(), SqlError> {
    connection.execute(META_SET, &[key.into(), value.into()])?;
    Ok(())
}

/// Entry count and total size per namespace of the active environment's
/// database, for reporting. Empty when the database isn't open.
pub(crate) fn summary() -> Vec<NamespaceSummary> {
    Database::open_active()
        .map(|database| database.summary())
        .unwrap_or_default()
}

/// The names of every namespace of the active environment's database. Empty
/// when the database isn't open.
pub(crate) fn namespaces() -> Vec<String> {
    Database::open_active()
        .map(|database| database.namespaces())
        .unwrap_or_default()
}

/// Deletes each namespace in `namespaces` from the active environment's database. Nothing when
/// the database isn't open.
pub(crate) fn purge_range(namespaces: Range<&str>) {
    if let Some(database) = Database::open_active() {
        database.purge_range(namespaces);
    }
}

/// Entry count and total size per namespace of the database behind
/// `connection`.
pub(crate) fn summarize(connection: &sql::Connection) -> Result<Vec<NamespaceSummary>, SqlError> {
    let mut summaries = Vec::new();
    connection.query(SUMMARY, &[], &mut |row| {
        summaries.push(NamespaceSummary {
            namespace: row.text(0)?,
            entries: row.integer(1)? as u64,
            bytes: row.integer(2)? as u64,
        });
        Ok::<_, SqlError>(())
    })?;
    Ok(summaries)
}

/// Restarts the log of the database behind `connection`, at `location`,
/// after a write that filled it: the frames it wrote are folded into the
/// file and the log starts again from its first frame.
///
/// Turso's own checkpoint after a commit folds the frames in but never
/// restarts the log, and it restarts on its own only at the start of a
/// write whose connection saw every frame folded in. A connection reopened
/// on the file as its only user forgets that it was, so a log nobody writes
/// again after one large write keeps its frames for good, and the next large
/// write adds to them past the frames turso's index holds between processes
/// (262,144), after which every read of the file fails as `database is
/// locked`. Restarting on the connection that wrote, before it is dropped,
/// is what keeps that from happening.
///
/// Best-effort: a reader holding the log keeps it from restarting, which is
/// logged, and the next write restarts it instead.
#[cfg(native_cache)]
fn restart_log(connection: &sql::Connection, location: &str) {
    match connection.checkpoint() {
        Ok(true) => {}
        Ok(false) => log::debug!("cubecl cache: the log of {location} is in use; not restarted"),
        Err(err) => log::warn!("cubecl cache: unable to restart the log of {location}: {err}"),
    }
}

/// The frames a log may hold since it last restarted before a write restarts
/// it first: a fifth of the 262,144 frames turso indexes between processes,
/// so that a write bounded as a replacement's batches are
/// ([`COPY_BATCH_BYTES`]) or an import's never brings the log near it.
#[cfg(native_cache)]
const LOG_RESTART_FRAMES: u64 = 50_000;

/// How long a write waits for the reads that keep a long log from
/// restarting before it is refused. A read is a lookup or a scan, which ends
/// in milliseconds.
#[cfg(native_cache)]
const LOG_RESTART_WAIT: core::time::Duration = core::time::Duration::from_secs(5);

/// Gives the log of the database behind `connection`, at `location`, room for
/// a write: when it holds [`LOG_RESTART_FRAMES`] or more since it last
/// restarted, restarts it ([`checkpoint`]), waiting up to
/// [`LOG_RESTART_WAIT`] for the reads that hold it.
///
/// The frames are counted from the size of the log file, whose frames are
/// `frame_bytes` each behind a 32-byte header, at no cost to a write: the file
/// only shrinks when the log restarts, so it counts at least what the log
/// holds.
///
/// # Errors
///
/// When the log can't be restarted in time, naming what holds it: writing on
/// would carry it toward what turso can index, past which every read of the
/// file fails.
#[cfg(native_cache)]
fn make_log_room(
    connection: &sql::Connection,
    location: &str,
    frame_bytes: u64,
) -> Result<(), String> {
    let frames = log_frames(location, frame_bytes);
    if sql::INDEXED_LOG_FRAMES.is_none() || frames < LOG_RESTART_FRAMES {
        return Ok(());
    }

    let started = std::time::Instant::now();
    let held = loop {
        let held = match connection.checkpoint() {
            Ok(true) => return Ok(()),
            Ok(false) | Err(SqlError::Busy(_)) => None,
            Err(err) => Some(err.to_string()),
        };
        if started.elapsed() >= LOG_RESTART_WAIT {
            break held;
        }
        std::thread::sleep(core::time::Duration::from_millis(50));
    };
    let cause = held.unwrap_or_else(|| {
        "a read held open on it, by another connection of this process or by another process, \
         keeps it from restarting"
            .to_string()
    });
    Err(format!(
        "the log of {location} holds {frames} frames since it last restarted, and it could not \
         be restarted within {LOG_RESTART_WAIT:?}: {cause}. The write was refused, since \
         writing on would carry the log toward the 262,144 frames the engine indexes between \
         processes, past which every read of the file fails. It succeeds once that read ends; \
         with no process using the file, moving its directory aside starts it afresh"
    ))
}

/// The frames the log of the database at `location` holds since it last
/// restarted, or more, counted from the size of its file.
#[cfg(native_cache)]
fn log_frames(location: &str, frame_bytes: u64) -> u64 {
    let bytes = std::fs::metadata(format!("{location}-wal")).map_or(0, |metadata| metadata.len());
    bytes.saturating_sub(32) / frame_bytes
}

/// The bytes one frame of the log takes: a page of the database behind
/// `connection` and the frame's 24-byte header.
fn frame_bytes(connection: &sql::Connection) -> Result<u64, SqlError> {
    // A row that doesn't decode leaves the default page size.
    let mut page_size = 4096;
    connection.pragma("page_size", &mut |row| {
        page_size = row.integer(0).unwrap_or(page_size);
    })?;
    Ok(page_size as u64 + 24)
}

/// The two header bytes that say which journal mode wrote the file, and the
/// legacy value: a rollback journal, which is what a reader expects of a file
/// with no `-wal` beside it.
#[cfg(native_cache)]
const JOURNAL_MODE_BYTES: core::ops::Range<u64> = 18..20;
#[cfg(native_cache)]
const LEGACY_JOURNAL: [u8; 2] = [1, 1];

/// Checkpoints the database behind `connection`, closes the connection and
/// marks the file at `path` a rollback-journal file, so that it stands on its
/// own and opens from a directory nobody may write.
///
/// The connection is taken so that it is closed before the header is
/// rewritten; nothing else in the process may hold the file open, or it writes
/// the WAL header back.
///
/// # Errors
///
/// When the checkpoint does not complete, or the header can't be written.
#[cfg(native_cache)]
pub(crate) fn make_standalone(
    connection: sql::Connection,
    path: &std::path::Path,
) -> Result<(), String> {
    if !connection.checkpoint().map_err(error)? {
        return Err(format!(
            "the WAL checkpoint of {} did not complete",
            path.display()
        ));
    }
    drop(connection);
    mark_rollback_journal(path).map_err(|err| err.to_string())
}

/// Marks a checkpointed database as a rollback-journal file, so that a reader
/// which cannot write beside it can still open it.
///
/// Turso is WAL-only: `PRAGMA journal_mode = DELETE` reports success and
/// changes nothing, so a file just checkpointed still carries a WAL header,
/// and `SQLite` keeps one too until its last connection closes.
/// `SQLite` takes that header at its word and insists on building the `-shm`
/// that WAL recovery needs — in a directory it may not write, the open fails
/// with "attempt to write a readonly database", which is exactly where a
/// shipped bundle lives.
///
/// After [`sql::Connection::checkpoint`] has folded and truncated the WAL, the file holds a
/// complete database and no WAL frames, so these two bytes are the only thing
/// distinguishing it from one a rollback-journal writer produced. Both
/// engines read the result: Turso ignores the field and opens its own WAL as
/// usual, and `SQLite` stops asking for a sidecar.
///
/// Call it on a checkpointed file with no connection open to it.
#[cfg(native_cache)]
fn mark_rollback_journal(path: &std::path::Path) -> std::io::Result<()> {
    use std::io::{Seek, SeekFrom, Write};

    let mut file = std::fs::OpenOptions::new().write(true).open(path)?;
    file.seek(SeekFrom::Start(JOURNAL_MODE_BYTES.start))?;
    file.write_all(&LEGACY_JOURNAL)?;
    file.sync_all()
}

/// How long a statement waits on a lock another process holds before it
/// reports [`Insertion::Failed`]. Several processes sharing a cache root is
/// routine; the wait is yield-based, so it costs the browser nothing it
/// can't afford.
const BUSY_TIMEOUT: core::time::Duration = core::time::Duration::from_secs(5);

/// A connection to `database`, with the busy timeout and `synchronous` set.
///
/// `synchronous` is per-connection and both engines default it to `FULL`,
/// which fsyncs on every commit. `NORMAL` may lose the last commits on a
/// power cut, which for a cache costs a recompute. It is set on read-only
/// connections too, where it is a no-op, so that every connection this module
/// hands out is configured alike.
pub(crate) fn connect(database: &sql::Database) -> Result<sql::Connection, SqlError> {
    let connection = database.connect()?;
    connection.busy_timeout(BUSY_TIMEOUT)?;
    connection.pragma("synchronous = NORMAL", &mut |_| {})?;
    Ok(connection)
}

fn origin_code(origin: Origin) -> i64 {
    match origin {
        Origin::Local => 0,
        Origin::Imported => 1,
    }
}

/// The inverse of [`origin_code`]. Only a local entry is `0`; anything else
/// was written by an import, the one other writer of the column.
fn origin_of(code: i64) -> Origin {
    match code {
        0 => Origin::Local,
        _ => Origin::Imported,
    }
}

fn error(error: SqlError) -> String {
    error.to_string()
}

/// `operation`'s result, with a panic inside the engine caught as a [`StorageFault`] rather than
/// left to unwind through the thread that called it, which may be a device's runner thread
/// a load is waiting on.
fn caught<T>(operation: impl FnOnce() -> Result<T, SqlError>) -> Result<T, StorageFault> {
    match std::panic::catch_unwind(core::panic::AssertUnwindSafe(operation)) {
        Ok(result) => result.map_err(StorageFault::Refused),
        Err(payload) => Err(StorageFault::Panicked(panic_message(payload.as_ref()))),
    }
}

/// The message a panic was raised with, when it is text.
fn panic_message(payload: &(dyn core::any::Any + Send)) -> String {
    match (
        payload.downcast_ref::<&str>(),
        payload.downcast_ref::<String>(),
    ) {
        (Some(message), _) => message.to_string(),
        (_, Some(message)) => message.clone(),
        _ => "a panic without a message".to_string(),
    }
}

/// Why a call on an environment's database failed.
enum StorageFault {
    /// The engine refused the call.
    Refused(SqlError),
    /// The engine panicked during the call, with this message.
    Panicked(String),
}

/// The cause of a [`StorageFault`], named for what fixes it.
enum FaultCause {
    /// The log outgrew the engine's shared frame index, so every read is refused as locked.
    IndexOverflowed,
    /// Another connection held the file past the busy timeout.
    Locked,
    /// The file is not a database, or is damaged.
    Damaged,
    /// The disk refused the write: full, or failing.
    Disk,
    /// The file was opened read-only.
    ReadOnly,
    /// The engine panicked.
    Panicked,
    /// Anything else, which the error's own words describe.
    Other,
}

impl StorageFault {
    fn cause(&self, location: &str) -> FaultCause {
        match self {
            Self::Refused(SqlError::Busy(_)) if index_overflowed(location) => {
                FaultCause::IndexOverflowed
            }
            Self::Refused(SqlError::Busy(_) | SqlError::HeldByAnotherProcess(_)) => {
                FaultCause::Locked
            }
            Self::Refused(SqlError::Corrupt(_) | SqlError::NotADatabase(_)) => FaultCause::Damaged,
            Self::Refused(SqlError::Full(_) | SqlError::Io(_)) => FaultCause::Disk,
            Self::Refused(SqlError::ReadOnly(_)) => FaultCause::ReadOnly,
            Self::Panicked(_) => FaultCause::Panicked,
            Self::Refused(SqlError::Other(_)) => FaultCause::Other,
        }
    }

    /// What failed in `location`, why, and what to do.
    fn explain(&self, location: &str) -> String {
        let said = match self {
            Self::Refused(error) => error.to_string(),
            Self::Panicked(message) => format!("{} panicked: {message}", sql::ENGINE),
        };
        let directory = std::path::Path::new(location)
            .parent()
            .map_or_else(|| location.to_string(), |dir| dir.display().to_string());
        match self.cause(location) {
            FaultCause::IndexOverflowed => format!(
                "{said}: the log of {location} holds more frames than the engine's shared index of \
                 262,144, so every read of it is refused. Once no process uses it, the next \
                 open restarts the log; until then, moving {directory} aside with no process \
                 using it starts afresh"
            ),
            FaultCause::Locked => format!(
                "{said}: another connection to {location}, of this process or another, held it \
                 past the {BUSY_TIMEOUT:?} busy timeout. It succeeds once that one finishes; if \
                 none is running, moving {directory} aside with no process using it starts \
                 afresh"
            ),
            FaultCause::Damaged => format!(
                "{said}: {location} is not a readable database. Moving {directory} aside with no \
                 process using it starts afresh, rebuilding what it held"
            ),
            FaultCause::Disk => format!(
                "{said}: the disk under {location} refused it. Free space there, or check the \
                 disk"
            ),
            FaultCause::ReadOnly => format!("{said}: {location} is open read-only"),
            FaultCause::Panicked => format!(
                "{said}, while working on {location}. The file may be in a state the engine does not \
                 expect; moving {directory} aside with no process using it starts afresh"
            ),
            FaultCause::Other => format!("{said} ({location})"),
        }
    }
}

/// Whether the shared frame index of `location`'s log has overflowed: its flag in the
/// `-tshm` header, which only Turso keeps. Read from the file, since Turso reports it only as
/// a lock.
fn index_overflowed(location: &str) -> bool {
    #[cfg(native_cache)]
    {
        use std::io::{Read, Seek, SeekFrom};
        const OVERFLOWED_AT: u64 = 44;

        let mut flag = [0u8; 1];
        std::fs::File::open(format!("{location}-tshm"))
            .and_then(|mut file| {
                file.seek(SeekFrom::Start(OVERFLOWED_AT))?;
                file.read_exact(&mut flag)
            })
            .is_ok_and(|()| flag != [0])
    }
    #[cfg(not(native_cache))]
    {
        let _ = location;
        false
    }
}

/// The storage serving `namespace` in the active environment's database.
pub fn open(namespace: &str) -> Result<Box<dyn Storage>, String> {
    DatabaseStorage::open(namespace.to_string())
        .map(|storage| Box::new(storage) as Box<dyn Storage>)
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
        let mut names = Vec::new();
        connection
            .query(
                "SELECT name FROM sqlite_schema WHERE type = 'table' ORDER BY name",
                &[],
                &mut |row| {
                    names.push(row.text(0)?);
                    Ok::<_, SqlError>(())
                },
            )
            .unwrap();
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
            connection.execute(CREATE_META, &[]).unwrap();
            meta_set(&connection, SCHEMA_VERSION_KEY, "999").unwrap();
            connection
                .execute(
                    "CREATE TABLE entries (store TEXT NOT NULL, key BLOB NOT NULL, \
                     value BLOB NOT NULL, PRIMARY KEY (store, key))",
                    &[],
                )
                .unwrap();
            connection
                .execute("INSERT INTO entries VALUES ('old', X'01', X'02')", &[])
                .unwrap();
        }

        let storage = DatabaseStorage::open("old".to_string()).unwrap();
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

    /// A file opened at its path, whichever environment is active, holds what
    /// was written to it with where each entry came from; read-only, it is
    /// read as it is, and making it standalone leaves a file a reader opens
    /// from where it cannot write.
    #[test_log::test]
    #[cfg_attr(miri, ignore)]
    fn a_database_opens_at_its_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("elsewhere.db");

        let written = Database::open_at(&path, Access::ReadWrite).unwrap();
        assert_eq!(
            written.insert("kernels", b"local", b"1", Origin::Local),
            Insertion::Stored
        );
        assert_eq!(
            written.insert("kernels", b"imported", b"22", Origin::Imported),
            Insertion::Stored
        );
        assert!(!written.is_standalone());
        written.into_standalone().unwrap();

        let read = Database::open_at(&path, Access::ReadOnly).unwrap();
        assert!(read.is_standalone());
        let mut entries = Vec::new();
        read.scan_with_origin("kernels", &mut |key, _, origin| {
            entries.push((key.to_vec(), origin));
        });
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(
            entries,
            vec![
                (b"imported".to_vec(), Origin::Imported),
                (b"local".to_vec(), Origin::Local),
            ]
        );
        assert_eq!(
            read.summary(),
            vec![NamespaceSummary {
                namespace: "kernels".to_string(),
                entries: 2,
                bytes: 16,
            }]
        );
    }

    /// Reading a file leaves it as it was: a standalone file read is still
    /// standalone, and a missing one is not created.
    #[test_log::test]
    #[cfg_attr(miri, ignore)]
    fn reading_a_file_leaves_it_as_it_was() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("standalone.db");
        let written = Database::open_at(&path, Access::ReadWrite).unwrap();
        written.insert("kernels", b"key", b"1", Origin::Local);
        written.into_standalone().unwrap();

        let read = Database::open_at(&path, Access::ReadOnly).unwrap();
        assert_eq!(
            read.get("kernels", b"key"),
            Some(Bytes::from_bytes_vec(vec![b'1']))
        );
        drop(read);
        assert!(
            Database::open_at(&path, Access::ReadOnly)
                .unwrap()
                .is_standalone()
        );

        let missing = dir.path().join("missing.db");
        assert!(Database::open_at(&missing, Access::ReadOnly).is_err());
        assert!(!missing.exists());
    }

    /// A read-only open never rebuilds what it reads, so it refuses a file it
    /// would misread: one of another schema, or one that is no environment.
    #[test_log::test]
    #[cfg_attr(miri, ignore)]
    fn a_read_only_open_refuses_what_it_would_misread() {
        let dir = tempfile::tempdir().unwrap();

        let other_schema = dir.path().join("other-schema.db");
        {
            let database = block_on(open_database(other_schema.to_str().unwrap())).unwrap();
            let connection = connect(&database).unwrap();
            connection.execute(CREATE_META, &[]).unwrap();
            meta_set(&connection, SCHEMA_VERSION_KEY, "999").unwrap();
        }
        assert!(Database::open_at(&other_schema, Access::ReadOnly).is_err());

        let foreign = dir.path().join("foreign.db");
        {
            let database = block_on(open_database(foreign.to_str().unwrap())).unwrap();
            let connection = connect(&database).unwrap();
            connection
                .execute("CREATE TABLE notes (body TEXT)", &[])
                .unwrap();
        }
        assert!(Database::open_at(&foreign, Access::ReadOnly).is_err());
        assert_eq!(
            tables(foreign.to_str().unwrap()),
            vec!["notes"],
            "nothing was created in the file it refused"
        );
    }

    /// A read-only handle on the active environment's file must not leave its
    /// stores unable to write: the engine hands every open of a file the object
    /// the first one configured, so a read-only first open would otherwise be
    /// the database the environment writes through.
    #[test_log::test]
    #[cfg_attr(miri, ignore)]
    #[serial_test::serial]
    fn a_read_only_handle_leaves_the_active_environment_writable() {
        let dir = tempfile::tempdir().unwrap();
        let path = std::path::PathBuf::from(active_location(dir.path()));
        drop(Database::open_at(&path, Access::ReadWrite).unwrap());

        let read = Database::open_at(&path, Access::ReadOnly).unwrap();
        let storage = DatabaseStorage::open("kernels".to_string()).unwrap();
        assert_eq!(
            storage.insert(
                b"while-read",
                Bytes::from_bytes_vec(b"1".to_vec()),
                Origin::Local
            ),
            Insertion::Stored
        );
        drop(read);
        assert_eq!(
            DatabaseStorage::open("kernels".to_string())
                .unwrap()
                .insert(
                    b"after",
                    Bytes::from_bytes_vec(b"2".to_vec()),
                    Origin::Local
                ),
            Insertion::Stored
        );
    }

    /// Only a handle that owns the file may make it standalone: a read-only
    /// one promised never to write it, and the active environment's database
    /// stays open behind its stores, which would write the WAL header back.
    #[test_log::test]
    #[cfg_attr(miri, ignore)]
    #[serial_test::serial]
    fn only_an_unshared_writable_file_is_made_standalone() {
        let dir = tempfile::tempdir().unwrap();

        let path = dir.path().join("elsewhere.db");
        drop(Database::open_at(&path, Access::ReadWrite).unwrap());
        let read = Database::open_at(&path, Access::ReadOnly).unwrap();
        assert!(read.into_standalone().is_err());

        let active = std::path::PathBuf::from(active_location(dir.path()));
        let written = Database::open_at(&active, Access::ReadWrite).unwrap();
        assert!(written.into_standalone().is_err());
    }

    /// A standalone file is still an environment: opened writable it takes
    /// writes, and the first one puts it back in WAL mode.
    #[test_log::test]
    #[cfg_attr(miri, ignore)]
    fn a_standalone_file_reopens_writable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("standalone.db");
        let written = Database::open_at(&path, Access::ReadWrite).unwrap();
        written.insert("kernels", b"first", b"1", Origin::Local);
        written.into_standalone().unwrap();

        let reopened = Database::open_at(&path, Access::ReadWrite).unwrap();
        assert_eq!(
            reopened.insert("kernels", b"second", b"2", Origin::Local),
            Insertion::Stored
        );
        assert!(!reopened.is_standalone());
        assert_eq!(reopened.namespaces(), vec!["kernels".to_string()]);
        assert_eq!(
            reopened.get("kernels", b"first"),
            Some(Bytes::from_bytes_vec(b"1".to_vec()))
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
            cached: false,
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

        let storage = DatabaseStorage::open("kept".to_string()).unwrap();
        assert_eq!(
            storage.insert(b"key", Bytes::from_bytes_vec(vec![7]), Origin::Local),
            Insertion::Stored
        );

        // The registry hands the same database back; go around it to make
        // `migrate` run again on the file as it is on disk.
        let database = block_on(open_database(&location)).unwrap();
        migrate(&database).unwrap();

        let reopened = DatabaseStorage::open("kept".to_string()).unwrap();
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

        let first = DatabaseStorage::open("namespace".to_string()).unwrap();
        let second = DatabaseStorage::open("namespace".to_string()).unwrap();

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

    #[test_log::test]
    #[cfg_attr(miri, ignore)]
    fn a_range_purge_takes_its_start_and_stops_before_its_end() {
        let dir = tempfile::tempdir().unwrap();
        let database = Database::open_at(&dir.path().join("range.db"), Access::ReadWrite).unwrap();
        for namespace in ["a/x", "b/", "b/y", "b0", "c/z"] {
            database.insert(namespace, b"key", b"value", Origin::Local);
        }

        database.purge_range("b/".."b0");

        assert_eq!(
            database.namespaces(),
            vec!["a/x".to_string(), "b0".to_string(), "c/z".to_string()]
        );
    }

    /// [`log_frames`](super::log_frames) of the database at `path`, at the
    /// default page size every test database has.
    pub(super) fn log_frames(path: &std::path::Path) -> u64 {
        super::log_frames(path.to_str().unwrap(), 4096 + 24)
    }

    /// `database` filled with `megabytes` of 64 KiB values in `namespace`.
    pub(super) fn fill(database: &Database, namespace: &str, megabytes: usize) {
        let value = Bytes::from_bytes_vec(vec![7u8; 64 * 1024]);
        let mut entries = (0..megabytes as u32 * 16).map(|index| {
            (
                Bytes::from_bytes_vec(index.to_le_bytes().to_vec()),
                value.clone(),
            )
        });
        let summary = database.insert_many(namespace, &mut entries, Origin::Local);
        assert_eq!(summary.failed, 0, "{summary:?}");
    }

    /// A replacement leaves the log it filled restarted: the copy writes
    /// about two frames a page, which turso folds into the file but never
    /// restarts on its own once this connection is gone.
    #[test_log::test]
    #[cfg_attr(miri, ignore)]
    fn a_replacement_leaves_its_log_restarted() {
        let dir = tempfile::tempdir().unwrap();
        let source = Database::open_at(&dir.path().join("saved.db"), Access::ReadWrite).unwrap();
        fill(&source, "kernels", 40);

        let path = dir.path().join("building.db");
        let target = Database::open_at(&path, Access::ReadWrite).unwrap();
        target.replace_with(Some(&source)).unwrap();

        assert_eq!(log_frames(&path), 0);
        assert_eq!(target.summary()[0].entries, 40 * 16);
    }

    /// The frames the log of the database at `path` holds since it last
    /// restarted, as turso's index between processes counts them: the
    /// `max_frame` of its `-tshm` header, 0 before there is one.
    #[cfg(turso_engine)]
    pub(super) fn indexed_frames(path: &std::path::Path) -> u64 {
        use std::io::{Read, Seek, SeekFrom};

        let mut tshm = std::ffi::OsString::from(path);
        tshm.push("-tshm");
        let mut header = [0u8; 8];
        let read = std::fs::File::open(&tshm).and_then(|mut file| {
            file.seek(SeekFrom::Start(56))?;
            file.read_exact(&mut header)
        });
        read.map_or(0, |()| u64::from_le_bytes(header))
    }

    /// A replacement copies in batches, so the log never holds more than a
    /// few of them: in one write, 256 MiB would be about 131,000 frames, half
    /// of what turso indexes, and a Flash-Next environment past all of it.
    #[test_log::test]
    #[cfg_attr(miri, ignore)]
    #[cfg(turso_engine)]
    fn a_replacement_keeps_its_log_short_while_it_copies() {
        let dir = tempfile::tempdir().unwrap();
        let source = Database::open_at(&dir.path().join("saved.db"), Access::ReadWrite).unwrap();
        fill(&source, "kernels", 256);

        let path = dir.path().join("building.db");
        let target = Database::open_at(&path, Access::ReadWrite).unwrap();
        let copying = std::sync::atomic::AtomicBool::new(true);
        let most = std::thread::scope(|scope| {
            let sampler = scope.spawn(|| {
                let mut most = 0;
                while copying.load(std::sync::atomic::Ordering::Relaxed) {
                    most = most.max(indexed_frames(&path));
                    std::thread::sleep(core::time::Duration::from_millis(2));
                }
                most
            });
            target.replace_with(Some(&source)).unwrap();
            copying.store(false, std::sync::atomic::Ordering::Relaxed);
            sampler.join().unwrap()
        });

        assert!(most > 0, "the sampler saw the copy");
        assert!(
            most <= 2 * (COPY_BATCH_BYTES as u64 / 4096) + 1_000,
            "the log held {most} frames at once"
        );
        assert_eq!(target.summary()[0].entries, 256 * 16);
    }

    /// A long log that a read held open keeps from restarting refuses a
    /// write, saying what holds it, rather than growing on toward what turso
    /// indexes; once the read ends, the write restarts the log and lands.
    ///
    /// The read is begun on frames not yet folded into the file: a write of
    /// 1 MiB, short of the 1,000 frames after which turso folds a commit in.
    /// A read begun once every frame is folded in reads the file, and holds
    /// nothing.
    #[test_log::test]
    #[cfg_attr(miri, ignore)]
    #[cfg(turso_engine)]
    fn a_held_read_keeps_a_long_log_from_growing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("building.db");
        let writer = Database::open_at(&path, Access::ReadWrite).unwrap();
        fill(&writer, "kernels", 1);

        let reader = Database::open_at(&path, Access::ReadWrite).unwrap();
        reader.with_connection(|connection| {
            connection.execute("BEGIN", &[]).unwrap();
            connection
                .query("SELECT count(*) FROM entries", &[], &mut |_| {
                    Ok::<_, SqlError>(())
                })
                .unwrap();
        });
        // 128 MiB more: about 65,000 frames, more than a write may start on.
        fill(&writer, "programs", 128);
        assert!(log_frames(&path) >= LOG_RESTART_FRAMES);

        let started = std::time::Instant::now();
        let refused = writer.insert("kernels", b"late", b"1", Origin::Local);
        let waited = started.elapsed();
        match refused {
            Insertion::Failed(reason) => {
                assert!(reason.contains("a read held open on it"), "{reason}");
                assert!(reason.contains("building.db"), "{reason}");
            }
            other => panic!("the write landed while the read held the log: {other:?}"),
        }
        assert!(waited >= LOG_RESTART_WAIT, "refused after {waited:?}");

        reader.with_connection(|connection| connection.execute("COMMIT", &[]).unwrap());
        assert_eq!(
            writer.insert("kernels", b"late", b"1", Origin::Local),
            Insertion::Stored
        );
        assert!(log_frames(&path) < 1_000, "{} frames", log_frames(&path));
    }

    /// A write another connection's lock keeps out fails naming the file, the lock and what to
    /// do, rather than turso's bare "database is locked".
    #[test]
    fn a_locked_write_names_the_file_the_cause_and_the_fix() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("building.db");
        let writer = Database::open_at(&path, Access::ReadWrite).unwrap();
        let holder = Database::open_at(&path, Access::ReadWrite).unwrap();
        holder.with_connection(|connection| {
            connection.execute("BEGIN IMMEDIATE", &[]).unwrap();
        });

        match writer.insert("kernels", b"key", b"1", Origin::Local) {
            Insertion::Failed(reason) => {
                assert!(reason.contains("building.db"), "{reason}");
                assert!(reason.contains("(kernels)"), "{reason}");
                assert!(reason.contains("busy timeout"), "{reason}");
                assert!(reason.contains("moving"), "{reason}");
            }
            other => panic!("the write landed while another connection held the lock: {other:?}"),
        }
        holder.with_connection(|connection| connection.execute("ROLLBACK", &[]).unwrap());
    }

    /// A lock turso reports over a log whose shared index overflowed is told as the overflow it
    /// is.
    #[test]
    fn a_lock_over_an_overflowed_index_is_told_as_the_overflow() {
        let dir = tempfile::tempdir().unwrap();
        let location = dir.path().join("building.db").to_str().unwrap().to_string();
        let locked = StorageFault::Refused(SqlError::Busy("database is locked".into()));
        assert!(matches!(locked.cause(&location), FaultCause::Locked));

        let mut header = vec![0u8; 128];
        header[44] = 1;
        std::fs::write(format!("{location}-tshm"), header).unwrap();
        assert!(matches!(
            locked.cause(&location),
            FaultCause::IndexOverflowed
        ));
        assert!(locked.explain(&location).contains("262,144"));
    }

    /// A panic inside the engine becomes the call's failure, with its message.
    #[test]
    fn a_panic_inside_the_storage_is_a_failure() {
        let fault = caught::<()>(|| panic!("reader slot released by non-owner")).unwrap_err();
        let explained = fault.explain("/cache/building.db");
        let panicked = format!(
            "{} panicked: reader slot released by non-owner",
            sql::ENGINE
        );
        assert!(explained.contains(&panicked), "{explained}");
        assert!(explained.contains("/cache/building.db"), "{explained}");
    }

    /// Where [`grow_a_log_until_killed`] writes, when a test runs it as a
    /// child process.
    #[cfg(turso_engine)]
    const GROW_LOG_AT: &str = "CUBECL_TEST_GROW_LOG_AT";

    /// The child process of
    /// [`a_log_a_killed_writer_left_past_the_index_reopens`]: holds a read on
    /// frames not folded in yet, so the log can neither restart nor be folded
    /// in, and commits 32 MiB at a time until it is killed, printing the
    /// frames the log holds after each commit. Does nothing unless the test
    /// runs it.
    #[test]
    #[ignore = "a child process of a_log_a_killed_writer_left_past_the_index_reopens"]
    #[cfg(turso_engine)]
    fn grow_a_log_until_killed() {
        let Some(location) = std::env::var_os(GROW_LOG_AT) else {
            return;
        };
        let location = location.into_string().unwrap();
        let database = block_on(sql::Database::open(&location, Opening::Shared)).unwrap();
        migrate(&database).unwrap();
        let writer = connect(&database).unwrap();
        let reader = connect(&database).unwrap();
        let value = vec![7u8; 64 * 1024];
        let seen = [
            "seen".into(),
            b"\0".as_slice().into(),
            (&value).into(),
            0.into(),
        ];
        writer.execute(INSERT, &seen).unwrap();
        reader.execute("BEGIN", &[]).unwrap();
        reader
            .query("SELECT count(*) FROM entries", &[], &mut |_| {
                Ok::<_, SqlError>(())
            })
            .unwrap();

        let page = frame_bytes(&writer).unwrap();
        // Once the log is past the index every statement fails as busy: the
        // writer keeps trying, so that only the kill ends it.
        let mut batch = 0u32;
        loop {
            let committed = (|| {
                let transaction = Transaction::new(&writer, write_transaction())?;
                for index in 0..512u32 {
                    let key = [batch.to_le_bytes(), index.to_le_bytes()].concat();
                    let params = ["kernels".into(), (&key).into(), (&value).into(), 0.into()];
                    transaction.execute(INSERT, &params)?;
                }
                transaction.commit()
            })();
            std::println!(
                "frames {} {committed:?}",
                super::log_frames(&location, page)
            );
            batch += 1;
        }
    }

    /// A writer killed with its log past the 262,144 frames turso indexes
    /// between processes, which a long read kept from restarting, leaves a
    /// file every read of which fails as `database is locked`: the next open
    /// folds the log in and reads within seconds.
    #[test_log::test]
    #[ignore = "writes past the frame index turso holds, about 2 GB over minutes"]
    #[cfg(all(unix, turso_engine))]
    fn a_log_a_killed_writer_left_past_the_index_reopens() {
        use std::io::BufRead;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("building.db");
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "persistence::database::tests::grow_a_log_until_killed",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(GROW_LOG_AT, &path)
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let lines = std::io::BufReader::new(child.stdout.take().unwrap()).lines();
        for line in lines {
            let line = line.unwrap();
            let frames: Option<u64> = line
                .strip_prefix("frames ")
                .and_then(|rest| rest.split(' ').next()?.parse().ok());
            // Past the index, a write fails, so the log stops just beyond it.
            if frames.is_some_and(|frames| frames > 262_144) {
                break;
            }
        }
        // SIGKILL: no unwinding, no close, no checkpoint.
        child.kill().unwrap();
        child.wait().unwrap();
        assert!(log_frames(&path) > 262_144, "{} frames", log_frames(&path));

        let started = std::time::Instant::now();
        let database = Database::open_at(&path, Access::ReadWrite).unwrap();
        assert!(database.get("seen", &[0u8]).is_some(), "the file reads");
        assert!(
            started.elapsed() < core::time::Duration::from_secs(120),
            "read after {:?}",
            started.elapsed()
        );
        assert!(log_frames(&path) < 1_000, "{} frames", log_frames(&path));
    }
}

#[cfg(all(
    test,
    native_cache,
    turso_engine,
    any(unix, windows),
    target_pointer_width = "64",
    not(miri)
))]
mod process_tests;
