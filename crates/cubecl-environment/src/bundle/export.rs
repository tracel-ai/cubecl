use std::path::{Path, PathBuf};
use std::string::{String, ToString};
use std::vec::Vec;

use crate::bytes::Bytes;
use crate::persistence::{Database, db_file_name};

use super::flat;
use super::{BundleError, BundleManifest, EnvironmentInfo, MANIFEST_SCHEMA};

/// Which on-disk layout [`export`] writes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BundleFormat {
    /// One `SQLite` file, read by [`SqliteBundle`](super::SqliteBundle). The
    /// native format: it needs a file system, but stays queryable.
    #[default]
    Sqlite,
    /// One flat blob, read by [`EmbeddedBundle`](super::EmbeddedBundle). The
    /// portable format: embed it with `include_bytes!` or fetch it at runtime
    /// on wasm and no-std targets, which have no file system to open.
    Flat,
}

/// Options for [`export`].
#[derive(Debug, Clone, Default)]
pub struct ExportOptions {
    /// Human-chosen bundle name, e.g. "H100 Linux".
    pub name: String,
    /// The environments the bundle was captured on. `os` and `arch` are
    /// auto-filled from the build target when left empty.
    pub environments: Vec<EnvironmentInfo>,
    /// Only export namespaces under one of these prefixes, e.g. `autotune`
    /// or `cuda`. A prefix matches whole segments, so it selects the
    /// namespace itself and everything below it. Empty means every namespace.
    pub namespaces: Vec<String>,
    /// The layout to write. Pick [`BundleFormat::Flat`] for wasm and no-std
    /// targets.
    pub format: BundleFormat,
}

/// Copies entries from one or more cache roots into a bundle file.
///
/// Merging several roots deduplicates by `(namespace, key)` rather than
/// concatenating bytes, and restricting the export to a few namespaces is a
/// filter. In [`BundleFormat::Sqlite`] both sides are `SQLite` databases with
/// the same `entries` table, so the whole export is one `INSERT ... SELECT`
/// across an attached source.
///
/// The typical workflow: run the application once so autotune and the
/// compilation caches are warm, then export the cache root.
pub fn export<R: AsRef<Path>, O: AsRef<Path>>(
    cache_roots: &[R],
    out: O,
    options: &ExportOptions,
) -> Result<BundleManifest, BundleError> {
    let out = out.as_ref();
    prepare_output(out)?;

    let manifest = BundleManifest {
        schema: MANIFEST_SCHEMA,
        name: options.name.clone(),
        cubecl_version: env!("CARGO_PKG_VERSION").to_string(),
        created_unix_secs: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()
            .map(|elapsed| elapsed.as_secs()),
        environments: resolve_environments(&options.environments),
    };

    let sources: Vec<PathBuf> = cache_roots
        .iter()
        .filter_map(|root| {
            let root = root.as_ref();
            source_database(root).or_else(|| {
                log::warn!("Bundle export: no cache database under {root:?}, skipping.");
                None
            })
        })
        .collect();

    let exported = match options.format {
        BundleFormat::Sqlite => export_sqlite(out, &sources, options, &manifest)?,
        BundleFormat::Flat => export_flat(out, &sources, options, &manifest)?,
    };

    if exported == 0 {
        log::warn!(
            "Bundle export: no entries matched. Run the application once so the caches \
             are warm, and check the namespace prefixes."
        );
    }

    Ok(manifest)
}

fn export_sqlite(
    out: &Path,
    sources: &[PathBuf],
    options: &ExportOptions,
    manifest: &BundleManifest,
) -> Result<usize, BundleError> {
    let database = Database::open(out, false)?;
    let mut exported = 0;

    for source in sources {
        exported += copy_entries(&database, source, &options.namespaces)?;
    }

    manifest.write(&database)?;

    Ok(exported)
}

fn export_flat(
    out: &Path,
    sources: &[PathBuf],
    options: &ExportOptions,
    manifest: &BundleManifest,
) -> Result<usize, BundleError> {
    let mut entries = flat::Entries::new();

    for source in sources {
        let database = Database::open(source, true)?;
        // First root wins on collision, matching `INSERT OR IGNORE`.
        read_entries(&database, &options.namespaces, &mut entries)?;
    }

    flat::write(out, &entries, manifest)?;

    Ok(entries.len())
}

/// Collects the requested namespaces of `database` into `entries`.
fn read_entries(
    database: &Database,
    namespaces: &[String],
    entries: &mut flat::Entries,
) -> Result<(), BundleError> {
    database.with_connection(|conn| {
        let mut statement = conn.prepare(
            "SELECT namespace, key, value FROM entries \
             WHERE ?1 = '' OR namespace = ?1 \
                OR substr(namespace, 1, length(?1) + 1) = ?1 || '/' \
             ORDER BY namespace, key",
        )?;

        // An empty filter selects everything, so the two cases share one query.
        let filters: Vec<&str> = if namespaces.is_empty() {
            std::vec![""]
        } else {
            namespaces.iter().map(String::as_str).collect()
        };

        for filter in filters {
            let mut rows = statement.query(rusqlite::params![filter])?;
            while let Some(row) = rows.next()? {
                // The driver hands back owned buffers; the value becomes
                // `Bytes` as it enters the map.
                let namespace: String = row.get(0)?;
                let key: Vec<u8> = row.get(1)?;
                let value = Bytes::from_bytes_vec(row.get(2)?);

                entries.entry((namespace, key)).or_insert(value);
            }
        }

        Ok::<_, rusqlite::Error>(())
    })?;

    Ok(())
}

/// The cache database of `root`, which may be a cache root directory or the
/// database file itself.
fn source_database(root: &Path) -> Option<PathBuf> {
    let path = if root.is_dir() {
        root.join(db_file_name(&crate::environment::active()))
    } else {
        root.to_path_buf()
    };

    path.is_file().then_some(path)
}

/// Attaches `source` and copies the requested namespaces into `database`.
fn copy_entries(
    database: &Database,
    source: &Path,
    namespaces: &[String],
) -> Result<usize, BundleError> {
    let source = source.to_string_lossy().to_string();

    let copied = database.with_connection(|conn| {
        conn.execute("ATTACH DATABASE ?1 AS source", rusqlite::params![source])?;

        let result = copy_attached(conn, namespaces);

        // Detach even when the copy failed, so the next root can attach.
        if let Err(err) = conn.execute("DETACH DATABASE source", []) {
            log::warn!("Bundle export: detaching {source:?} failed: {err}");
        }

        result
    })?;

    Ok(copied)
}

fn copy_attached(
    conn: &rusqlite::Connection,
    namespaces: &[String],
) -> Result<usize, rusqlite::Error> {
    // `INSERT OR IGNORE` is what makes merging several roots safe: the
    // (namespace, key) primary key collapses duplicates instead of appending them
    // twice, and an entry already exported from an earlier root wins.
    // The origin column is written explicitly: `INSERT OR IGNORE` treats a
    // NOT NULL violation as a row to skip, so omitting it would silently copy
    // nothing. Shipped rows are marked imported, which is what they become.
    const ALL: &str = "INSERT OR IGNORE INTO main.entries (namespace, key, value, origin) \
                       SELECT namespace, key, value, 1 FROM source.entries";
    // A plain prefix match on whole segments, avoiding LIKE's wildcards.
    const FILTERED: &str = "INSERT OR IGNORE INTO main.entries (namespace, key, value, origin) \
                            SELECT namespace, key, value, 1 FROM source.entries \
                            WHERE namespace = ?1 \
                               OR substr(namespace, 1, length(?1) + 1) = ?1 || '/'";

    if namespaces.is_empty() {
        return conn.execute(ALL, []);
    }

    let mut copied = 0;
    for namespace in namespaces {
        copied += conn.execute(FILTERED, rusqlite::params![namespace])?;
    }

    Ok(copied)
}

/// Makes sure `out` is a bundle file we may write.
///
/// An existing bundle is replaced, so re-exporting never merges into a stale
/// snapshot. Any other existing file is left alone and reported.
fn prepare_output(out: &Path) -> Result<(), BundleError> {
    if let Some(parent) = out.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }

    if !out.exists() {
        return Ok(());
    }

    match Database::open(out, true)
        .map_err(BundleError::from)
        .and_then(|database| BundleManifest::read(&database))
    {
        Ok(manifest) => {
            log::info!(
                "Replacing the existing bundle '{}' at {out:?}",
                manifest.name
            );
        }
        Err(_) => {
            return Err(BundleError::Io(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                std::format!("{out:?} exists and is not a cubecl bundle; remove it first"),
            )));
        }
    }

    std::fs::remove_file(out)?;
    // WAL sidecars of the replaced bundle would otherwise be applied to the
    // new file.
    for suffix in ["-wal", "-shm"] {
        let sidecar = PathBuf::from(std::format!("{}{suffix}", out.display()));
        if sidecar.exists() {
            std::fs::remove_file(sidecar)?;
        }
    }

    Ok(())
}

fn resolve_environments(configured: &[EnvironmentInfo]) -> Vec<EnvironmentInfo> {
    let mut environments = configured.to_vec();
    if environments.is_empty() {
        environments.push(EnvironmentInfo::default());
    }

    for environment in &mut environments {
        if environment.os.is_empty() {
            environment.os = std::env::consts::OS.to_string();
        }
        if environment.arch.is_empty() {
            environment.arch = std::env::consts::ARCH.to_string();
        }
    }

    environments
}
