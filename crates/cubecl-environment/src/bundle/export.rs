use std::path::{Path, PathBuf};
use std::string::{String, ToString};
use std::vec::Vec;

use crate::persistence::{DB_FILE_NAME, Database};

use super::{BundleError, BundleManifest, EnvironmentInfo, MANIFEST_SCHEMA};

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
}

/// Copies entries from one or more cache roots into a bundle file.
///
/// Both the cache and the bundle are `SQLite` databases with the same `entries`
/// table, so exporting is an `INSERT ... SELECT` across an attached source:
/// merging several roots dedupes on the primary key instead of concatenating
/// bytes, and restricting the export to a few namespaces is a `WHERE` clause.
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

    let database = Database::open(out, false)?;
    let mut exported = 0usize;

    for root in cache_roots {
        let Some(source) = source_database(root.as_ref()) else {
            log::warn!(
                "Bundle export: no cache database under {:?}, skipping.",
                root.as_ref()
            );
            continue;
        };

        exported += copy_entries(&database, &source, &options.namespaces)?;
    }

    if exported == 0 {
        log::warn!(
            "Bundle export: no entries matched. Run the application once so the caches \
             are warm, and check the namespace prefixes."
        );
    }

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
    manifest.write(&database)?;

    Ok(manifest)
}

/// The cache database of `root`, which may be a cache root directory or the
/// database file itself.
fn source_database(root: &Path) -> Option<PathBuf> {
    let path = if root.is_dir() {
        root.join(DB_FILE_NAME)
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
    const ALL: &str = "INSERT OR IGNORE INTO main.entries (namespace, key, value) \
                       SELECT namespace, key, value FROM source.entries";
    // A plain prefix match on whole segments, avoiding LIKE's wildcards.
    const FILTERED: &str = "INSERT OR IGNORE INTO main.entries (namespace, key, value) \
                            SELECT namespace, key, value FROM source.entries \
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
