use alloc::string::{String, ToString};
use alloc::vec::Vec;
use std::path::{Path, PathBuf};

use crate::bytes::Bytes;
use crate::future::block_on;
use crate::persistence::NamespaceSummary;
use crate::persistence::database::{self, SCHEMA_VERSION, SCHEMA_VERSION_KEY, connect};
use crate::persistence::sql::{self, Opening, SqlError};

use super::{Bundle, BundleError, BundleManifest};

/// The bytes every `SQLite` file starts with.
const MAGIC: &[u8; 16] = b"SQLite format 3\0";

/// A bundle stored as a single `SQLite` file, the format produced by
/// [`export`](super::export) on native targets.
///
/// The file is an environment database with a manifest in its `meta` table,
/// so besides being imported it can be mounted in place with
/// [`environment::load`](crate::environment::load). It is opened read-only:
/// a shipped bundle sits wherever it was installed, often somewhere nobody
/// may write.
///
#[derive(Debug)]
pub struct SqliteBundle {
    database: sql::Database,
    manifest: BundleManifest,
    path: PathBuf,
}

impl SqliteBundle {
    /// Opens a bundle file read-only, reading and validating its manifest.
    ///
    /// A cubecl version mismatch is not an error: the bundle still installs,
    /// its entries are simply never looked up, because the cubecl version is
    /// part of every namespace. A clear warning is logged instead of silent
    /// emptiness. A *schema* version this build doesn't read is an error: the
    /// rows might not decode at all.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, BundleError> {
        let path = path.as_ref();

        // Decided from the header before the engine sees the file, so a
        // foreign file reads as "not a bundle" rather than as whatever the
        // engine makes of it.
        if !is_sqlite(path)? {
            return Err(BundleError::NotABundle);
        }
        let location = path.to_str().ok_or_else(|| {
            BundleError::Storage(std::format!("bundle path {path:?} is not valid UTF-8"))
        })?;

        let database = block_on(sql::Database::open(location, Opening::ReadOnly))?;
        let connection = connect(&database)?;

        // "Not a bundle" is a narrow condition: a database without a `meta`
        // table, or with no version in it. Anything else — a locked, corrupt,
        // or unreadable database — is a real failure and must surface as such
        // rather than as the misleading "the file carries no bundle manifest".
        let expected = SCHEMA_VERSION.to_string();
        match database::meta_get(&connection, SCHEMA_VERSION_KEY).map_err(missing_meta)? {
            Some(found) if found == expected => {}
            Some(found) => return Err(BundleError::UnsupportedDatabase(found)),
            // A database with no version is not one of ours.
            None => return Err(BundleError::NotABundle),
        }

        let manifest = BundleManifest::read(&connection)?;
        manifest.warn_on_version_mismatch();

        Ok(Self {
            database,
            manifest,
            path: path.to_path_buf(),
        })
    }

    /// The bundle manifest.
    pub fn manifest(&self) -> &BundleManifest {
        &self.manifest
    }

    /// Entry count and total size per namespace, for reporting: what the
    /// file holds, without reading any of it.
    pub fn summary(&self) -> Vec<NamespaceSummary> {
        self.read("summarize", database::summarize)
            .unwrap_or_default()
    }

    /// Runs `read` on a fresh connection, logging a failure under `name`.
    fn read<T>(
        &self,
        name: &str,
        read: impl FnOnce(&sql::Connection) -> Result<T, SqlError>,
    ) -> Option<T> {
        connect(&self.database)
            .and_then(|connection| read(&connection))
            .inspect_err(|err| log::warn!("Unable to {name} {}: {err}", self.describe()))
            .ok()
    }
}

impl Bundle for SqliteBundle {
    fn get(&self, namespace: &str, key: &[u8]) -> Option<Bytes> {
        self.read("read", |connection| {
            let mut value = None;
            connection.query(
                "SELECT value FROM entries WHERE namespace = ?1 AND key = ?2",
                &[namespace.into(), key.into()],
                &mut |row| {
                    // A database row is materialized by the engine, so there
                    // is nothing to serve a zero-copy window into.
                    value = Some(Bytes::from_bytes_vec(row.blob(0)?));
                    Ok::<_, SqlError>(())
                },
            )?;
            Ok(value)
        })
        .flatten()
    }

    fn scan(&self, namespace: &str, visit: &mut dyn FnMut(&[u8], &[u8])) {
        self.read("scan", |connection| {
            connection.query(
                "SELECT key, value FROM entries WHERE namespace = ?1",
                &[namespace.into()],
                &mut |row| {
                    visit(&row.blob(0)?, &row.blob(1)?);
                    Ok::<_, SqlError>(())
                },
            )
        });
    }

    fn namespaces(&self) -> Vec<String> {
        self.read("list", |connection| {
            let mut namespaces = Vec::new();
            connection.query(
                "SELECT DISTINCT namespace FROM entries ORDER BY namespace",
                &[],
                &mut |row| {
                    namespaces.push(row.text(0)?);
                    Ok::<_, SqlError>(())
                },
            )?;
            Ok(namespaces)
        })
        .unwrap_or_default()
    }

    fn describe(&self) -> String {
        alloc::format!("bundle '{}' at {:?}", self.manifest.name, self.path)
    }
}

/// Whether `err` means "this file is not a cubecl bundle" rather than a genuine
/// database failure: a database without the `meta` table fails with a "no such
/// table" message.
pub(super) fn missing_meta(err: SqlError) -> BundleError {
    let message = err.to_string();
    if message.contains("no such table") {
        BundleError::NotABundle
    } else {
        BundleError::Storage(message)
    }
}

/// Whether `path` starts like a `SQLite` file.
///
/// A file too short to hold the header is not one. Any other failure to read
/// it is a real error and surfaces as such rather than as "not a bundle".
fn is_sqlite(path: &Path) -> Result<bool, BundleError> {
    use std::io::Read;

    let mut header = [0u8; 16];
    match std::fs::File::open(path)?.read_exact(&mut header) {
        Ok(()) => Ok(&header == MAGIC),
        Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => Ok(false),
        Err(err) => Err(err.into()),
    }
}
