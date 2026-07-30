use std::path::{Path, PathBuf};
use std::string::{String, ToString};
use std::vec::Vec;

use crate::bytes::Bytes;
use crate::persistence::{Database, db_file_name};

use super::flat;
use super::{BundleError, BundleManifest, EnvironmentInfo, MANIFEST_SCHEMA, flat_bundle_version};

/// A plain prefix match on whole segments, avoiding LIKE's wildcards.
///
/// Both formats select their rows with it, so restricting an export picks the
/// same namespaces whichever layout is written.
const NAMESPACE_PREFIX: &str = "namespace = ?1 OR substr(namespace, 1, length(?1) + 1) = ?1 || '/'";

/// Files `SQLite` writes next to a database, which belong to it.
const SIDECARS: [&str; 2] = ["-wal", "-shm"];

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
    /// namespace itself and everything below it. No prefix, or an empty one,
    /// means every namespace.
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
///
/// The bundle is built next to `out` and renamed onto it once complete, so a
/// failed export leaves the previous bundle, or no file at all, rather than a
/// truncated one.
pub fn export<R: AsRef<Path>, O: AsRef<Path>>(
    cache_roots: &[R],
    out: O,
    options: &ExportOptions,
) -> Result<BundleManifest, BundleError> {
    let out = out.as_ref();
    prepare_output(out, options.format)?;

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

    // Nothing writes to `out` until the bundle is complete, so an interrupted
    // export can't leave a file that later exports refuse to overwrite.
    let staged = staging_path(out);
    discard(&staged);

    let namespaces = filters(&options.namespaces);
    let exported = match options.format {
        BundleFormat::Sqlite => export_sqlite(&staged, &sources, namespaces, &manifest),
        BundleFormat::Flat => export_flat(&staged, &sources, namespaces, &manifest),
    };
    let exported = match exported {
        Ok(exported) => exported,
        Err(err) => {
            discard(&staged);
            return Err(err);
        }
    };

    publish(&staged, out)?;

    if exported == 0 {
        log::warn!(
            "Bundle export: no entries matched. Run the application once so the caches \
             are warm, and check the namespace prefixes."
        );
    }

    Ok(manifest)
}

/// The namespace prefixes to export, or `None` for every namespace.
///
/// An empty prefix selects everything, so a list holding one collapses to no
/// filter at all: both formats then take the same unfiltered path, instead of
/// one exporting everything and the other nothing.
fn filters(namespaces: &[String]) -> Option<&[String]> {
    let unrestricted = namespaces.is_empty() || namespaces.iter().any(String::is_empty);

    (!unrestricted).then_some(namespaces)
}

fn export_sqlite(
    out: &Path,
    sources: &[PathBuf],
    namespaces: Option<&[String]>,
    manifest: &BundleManifest,
) -> Result<usize, BundleError> {
    let database = Database::open(out, false)?;
    let mut exported = 0;

    for source in sources {
        exported += copy_entries(&database, source, namespaces)?;
    }

    manifest.write(&database)?;
    // A shipped bundle is read from wherever it was installed, which is often a
    // read-only directory. WAL is persistent in the file header and reading it
    // needs to create the wal-index next to the file, so leave the journal mode
    // where any reader can open it.
    database.finalize_for_shipping()?;

    Ok(exported)
}

fn export_flat(
    out: &Path,
    sources: &[PathBuf],
    namespaces: Option<&[String]>,
    manifest: &BundleManifest,
) -> Result<usize, BundleError> {
    let mut entries = flat::Entries::new();

    for source in sources {
        let database = Database::open(source, true)?;
        // First root wins on collision, matching `INSERT OR IGNORE`.
        read_entries(&database, namespaces, &mut entries)?;
    }

    flat::write(out, &entries, manifest)?;

    Ok(entries.len())
}

/// Collects the requested namespaces of `database` into `entries`.
fn read_entries(
    database: &Database,
    namespaces: Option<&[String]>,
    entries: &mut flat::Entries,
) -> Result<(), BundleError> {
    const SELECT: &str = "SELECT namespace, key, value FROM entries";

    database.with_connection(|conn| {
        let mut collect = |rows: &mut rusqlite::Rows<'_>| -> Result<(), rusqlite::Error> {
            while let Some(row) = rows.next()? {
                // The driver hands back owned buffers; the value becomes
                // `Bytes` as it enters the map.
                let namespace: String = row.get(0)?;
                let key: Vec<u8> = row.get(1)?;
                let value = Bytes::from_bytes_vec(row.get(2)?);

                entries.entry((namespace, key)).or_insert(value);
            }

            Ok(())
        };

        match namespaces {
            None => collect(&mut conn.prepare(SELECT)?.query([])?),
            Some(namespaces) => {
                let mut statement =
                    conn.prepare(&std::format!("{SELECT} WHERE {NAMESPACE_PREFIX}"))?;
                for namespace in namespaces {
                    collect(&mut statement.query(rusqlite::params![namespace])?)?;
                }

                Ok(())
            }
        }
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
    namespaces: Option<&[String]>,
) -> Result<usize, BundleError> {
    // ATTACH takes a string, and the connection is read-write, so SQLite would
    // create an empty database from a mangled path instead of failing: an
    // export that silently copies nothing. A path we can't pass losslessly is
    // reported rather than approximated.
    //
    // The source is attached read-write, not read-only immutable: a live cache
    // keeps its most recent entries in the WAL until a checkpoint, and only a
    // connection that opens the wal-index sees them. That does leave a `-wal`/
    // `-shm` sidecar next to the source for the duration, which is the normal
    // cost of reading a WAL database consistently.
    let source = source.to_str().ok_or_else(|| {
        BundleError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            std::format!("{source:?} is not valid UTF-8, so SQLite can't attach it"),
        ))
    })?;

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
    namespaces: Option<&[String]>,
) -> Result<usize, rusqlite::Error> {
    // `INSERT OR IGNORE` is what makes merging several roots safe: the
    // (namespace, key) primary key collapses duplicates instead of appending them
    // twice, and an entry already exported from an earlier root wins.
    // The origin column is written explicitly: `INSERT OR IGNORE` treats a
    // NOT NULL violation as a row to skip, so omitting it would silently copy
    // nothing. Shipped rows are marked imported, which is what they become.
    const COPY: &str = "INSERT OR IGNORE INTO main.entries (namespace, key, value, origin) \
                        SELECT namespace, key, value, 1 FROM source.entries";

    let Some(namespaces) = namespaces else {
        return conn.execute(COPY, []);
    };

    let filtered = std::format!("{COPY} WHERE {NAMESPACE_PREFIX}");
    let mut copied = 0;
    for namespace in namespaces {
        copied += conn.execute(&filtered, rusqlite::params![namespace])?;
    }

    Ok(copied)
}

/// Makes sure `out` is a bundle file we may write.
///
/// An existing bundle is replaced, so re-exporting never merges into a stale
/// snapshot. Any other existing file is left alone and reported.
fn prepare_output(out: &Path, format: BundleFormat) -> Result<(), BundleError> {
    if let Some(parent) = out.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }

    if !out.exists() {
        return Ok(());
    }

    // What makes a file ours depends on the layout being written: a flat blob
    // is identified by its header and could never answer as a database.
    let existing = match format {
        BundleFormat::Sqlite => Database::open(out, true)
            .map_err(BundleError::from)
            .and_then(|database| BundleManifest::read(&database))
            .map(|manifest| std::format!("'{}'", manifest.name))
            .ok(),
        // The flat manifest lives inside the blob, so the header identifies the
        // file; reading all of it just to name it isn't worth it.
        BundleFormat::Flat => flat_header(out).map(|version| std::format!("(flat v{version})")),
    };

    match existing {
        Some(described) => log::info!("Replacing the existing bundle {described} at {out:?}"),
        None => {
            return Err(BundleError::Io(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                std::format!("{out:?} exists and is not a cubecl bundle; remove it first"),
            )));
        }
    }

    Ok(())
}

/// The flat layout version `out` declares, if it is a flat bundle at all.
fn flat_header(out: &Path) -> Option<u32> {
    use std::io::Read;

    let mut header = [0u8; 12];
    let mut file = std::fs::File::open(out).ok()?;
    file.read_exact(&mut header).ok()?;

    flat_bundle_version(&header)
}

/// Where a bundle is built before it takes the place of `out`.
fn staging_path(out: &Path) -> PathBuf {
    PathBuf::from(std::format!("{}.tmp", out.display()))
}

fn sidecar(path: &Path, suffix: &str) -> PathBuf {
    PathBuf::from(std::format!("{}{suffix}", path.display()))
}

/// Removes a staged bundle and whatever `SQLite` left beside it.
fn discard(staged: &Path) {
    for path in [staged.to_path_buf()]
        .into_iter()
        .chain(SIDECARS.iter().map(|suffix| sidecar(staged, suffix)))
    {
        if path.exists()
            && let Err(err) = std::fs::remove_file(&path)
        {
            log::warn!("Bundle export: can't remove {path:?}: {err}");
        }
    }
}

/// Moves the finished bundle onto `out`, replacing what was there.
///
/// The sidecars are handled in the same move: a `-wal` left from the previous
/// bundle would otherwise be replayed into the new one.
fn publish(staged: &Path, out: &Path) -> Result<(), BundleError> {
    for suffix in SIDECARS {
        let stale = sidecar(out, suffix);
        if stale.exists() {
            std::fs::remove_file(stale)?;
        }
    }

    std::fs::rename(staged, out)?;

    for suffix in SIDECARS {
        let staged = sidecar(staged, suffix);
        if staged.exists() {
            std::fs::rename(staged, sidecar(out, suffix))?;
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
