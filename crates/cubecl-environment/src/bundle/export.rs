use std::collections::btree_map::Entry;
use std::path::{Path, PathBuf};
use std::string::{String, ToString};
use std::vec::Vec;

use crate::bytes::Bytes;
use crate::future::block_on;
use crate::persistence::turso::{self as database, connect};

use super::flat;
use super::{
    BundleError, BundleManifest, EnvironmentInfo, MANIFEST_SCHEMA, SqliteBundle,
    flat_bundle_version,
};

const SELECT: &str = "SELECT namespace, key, value FROM entries";
/// A plain prefix match on whole segments, avoiding LIKE's wildcards.
///
/// Both formats select their rows with it, so restricting an export picks the
/// same namespaces whichever layout is written.
const NAMESPACE_PREFIX: &str = "namespace = ?1 OR substr(namespace, 1, length(?1) + 1) = ?1 || '/'";

/// `ON CONFLICT DO NOTHING` is what makes merging several roots safe: the
/// (namespace, key) primary key collapses duplicates instead of appending them
/// twice, and an entry already exported from an earlier root wins. Shipped
/// rows are marked imported, which is what they become.
const INSERT: &str = "INSERT INTO entries (namespace, key, value, origin) \
                      VALUES (?1, ?2, ?3, 1) ON CONFLICT DO NOTHING";

/// Files `SQLite` writes next to a database, which belong to it.
const SIDECARS: [&str; 2] = ["-wal", "-shm"];

/// Which on-disk layout [`export`] writes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BundleFormat {
    /// One `SQLite` file, read by [`SqliteBundle`](super::SqliteBundle). The
    /// native format: it needs a file system, but stays queryable, and
    /// [`environment::load`](crate::environment::load) mounts it in place.
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
    /// Leave out the namespaces under any of these prefixes, applied after
    /// [`namespaces`](Self::namespaces): `records` exports an environment
    /// without the account of how it was built, for distribution.
    pub excluded_namespaces: Vec<String>,
    /// The layout to write. Pick [`BundleFormat::Flat`] for wasm and no-std
    /// targets.
    pub format: BundleFormat,
}

/// Copies entries from one or more cache roots into a bundle file.
///
/// Merging several roots deduplicates by `(namespace, key)` rather than
/// concatenating bytes, and restricting the export to a few namespaces is a
/// filter. The rows stream from each source into the bundle: a
/// [`BundleFormat::Sqlite`] export never holds more than one row in memory,
/// while the flat format is assembled in memory before it is written.
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
    let excluded = &options.excluded_namespaces;
    let exported = match options.format {
        BundleFormat::Sqlite => export_sqlite(&staged, &sources, namespaces, excluded, &manifest),
        BundleFormat::Flat => export_flat(&staged, &sources, namespaces, excluded, &manifest),
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

/// Whether `namespace` is `prefix` or below it, matching whole segments as
/// [`NAMESPACE_PREFIX`] does.
fn under(namespace: &str, prefix: &str) -> bool {
    namespace
        .strip_prefix(prefix)
        .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
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

/// Writes the sources into a database at `out` and leaves it standing on its
/// own: one file, checkpointed, that any reader can open.
fn export_sqlite(
    out: &Path,
    sources: &[PathBuf],
    namespaces: Option<&[String]>,
    excluded: &[String],
    manifest: &BundleManifest,
) -> Result<usize, BundleError> {
    let location = location(out)?;
    // Compaction is behind a flag in the engine; the staged file is the one
    // place a database of ours is ever compacted.
    let target = block_on(
        turso::Builder::new_local(location)
            .experimental_vacuum(true)
            .build(),
    )
    .map_err(storage_error)?;
    // The bundle carries the environment's own schema and version, so
    // `environment::load` can mount it as it would any environment file.
    database::migrate(&target).map_err(storage_error)?;

    let mut connection = connect(&target).map_err(storage_error)?;
    manifest.write(&connection)?;

    // One transaction for the whole copy: a row per statement would be a
    // WAL frame per row, and a partial bundle is not a bundle.
    let transaction = block_on(
        connection.transaction_with_behavior(turso::transaction::TransactionBehavior::Immediate),
    )
    .map_err(storage_error)?;
    let mut exported = 0;
    for source in sources {
        exported += read_entries(source, namespaces, &mut Sink::Sqlite(&transaction))?;
    }
    block_on(transaction.commit()).map_err(storage_error)?;

    // Copied, then deleted: the exclusion stays one statement per prefix
    // beside the copy's own filter, and the staged file is compacted after,
    // so what is left out costs nothing in the shipped one. Before the
    // checkpoint, which is what folds the deletes into the file.
    let delete = std::format!("DELETE FROM entries WHERE {NAMESPACE_PREFIX}");
    let mut removed = 0;
    for prefix in excluded.iter().filter(|prefix| !prefix.is_empty()) {
        removed += block_on(connection.execute(&delete, (prefix.as_str(),)))
            .map_err(storage_error)? as usize;
    }
    if removed > 0 {
        block_on(connection.execute("VACUUM", ())).map_err(storage_error)?;
    }
    exported -= removed;

    // A shipped bundle is read from wherever it was installed, which is often
    // a read-only directory, and the engine never checkpoints on close: a
    // file copied without its `-wal` is a file missing every row. The
    // connection is the last holder of the database once `target` is gone,
    // so closing it closes the file before its header is rewritten.
    drop(target);
    database::make_standalone(connection, out).map_err(BundleError::Storage)?;

    Ok(exported)
}

fn export_flat(
    out: &Path,
    sources: &[PathBuf],
    namespaces: Option<&[String]>,
    excluded: &[String],
    manifest: &BundleManifest,
) -> Result<usize, BundleError> {
    let mut entries = flat::Entries::new();

    for source in sources {
        read_entries(source, namespaces, &mut Sink::Flat(&mut entries))?;
    }
    entries.retain(|(namespace, _), _| {
        !excluded
            .iter()
            .any(|prefix| !prefix.is_empty() && under(namespace, prefix))
    });

    flat::write(out, &entries, manifest)?;

    Ok(entries.len())
}

/// Where the rows of a source go.
enum Sink<'a> {
    /// Into a database being written. Counts the rows it accepted, so a key
    /// an earlier root already exported is not counted twice.
    Sqlite(&'a turso::Connection),
    /// Into the map the flat format is assembled from. First root wins on
    /// collision, matching the database's `ON CONFLICT DO NOTHING`.
    Flat(&'a mut flat::Entries),
}

impl Sink<'_> {
    fn push(
        &mut self,
        namespace: String,
        key: Vec<u8>,
        value: Vec<u8>,
    ) -> Result<usize, BundleError> {
        match self {
            Sink::Sqlite(connection) => {
                let inserted = block_on(connection.execute(INSERT, (namespace, key, value)))
                    .map_err(storage_error)?;
                Ok(inserted as usize)
            }
            Sink::Flat(entries) => match entries.entry((namespace, key)) {
                Entry::Vacant(vacant) => {
                    vacant.insert(Bytes::from_bytes_vec(value));
                    Ok(1)
                }
                Entry::Occupied(_) => Ok(0),
            },
        }
    }
}

/// Streams the requested namespaces of the database at `source` into `sink`,
/// returning how many rows the sink accepted.
///
/// The source is opened read-only: a root may live where nobody can write —
/// a mounted bundle, a store path — and a read-only open still sees what a
/// live cache keeps in its WAL until the next checkpoint.
fn read_entries(
    source: &Path,
    namespaces: Option<&[String]>,
    sink: &mut Sink<'_>,
) -> Result<usize, BundleError> {
    let location = location(source)?;
    let database = block_on(turso::Builder::new_local(location).read_only(true).build())
        .map_err(storage_error)?;
    let connection = connect(&database).map_err(storage_error)?;

    match namespaces {
        None => {
            let rows = block_on(connection.query(SELECT, ())).map_err(storage_error)?;
            collect(rows, sink)
        }
        Some(namespaces) => {
            let query = std::format!("{SELECT} WHERE {NAMESPACE_PREFIX}");
            let mut accepted = 0;
            for namespace in namespaces {
                let rows = block_on(connection.query(&query, (namespace.as_str(),)))
                    .map_err(storage_error)?;
                accepted += collect(rows, sink)?;
            }
            Ok(accepted)
        }
    }
}

fn collect(mut rows: turso::Rows, sink: &mut Sink<'_>) -> Result<usize, BundleError> {
    let mut accepted = 0;
    while let Some(row) = block_on(rows.next()).map_err(storage_error)? {
        let namespace: String = row.get(0).map_err(storage_error)?;
        let key: Vec<u8> = row.get(1).map_err(storage_error)?;
        let value: Vec<u8> = row.get(2).map_err(storage_error)?;
        accepted += sink.push(namespace, key, value)?;
    }
    Ok(accepted)
}

/// The cache database of `root`, which may be a cache root directory or the
/// database file itself.
fn source_database(root: &Path) -> Option<PathBuf> {
    let path = if root.is_dir() {
        root.join(crate::environment::file_name(&crate::environment::active()))
    } else {
        root.to_path_buf()
    };

    path.is_file().then_some(path)
}

/// The engine takes a string, so a path it can't be given losslessly is
/// reported rather than approximated.
fn location(path: &Path) -> Result<&str, BundleError> {
    path.to_str()
        .ok_or_else(|| BundleError::Storage(std::format!("cache path {path:?} is not valid UTF-8")))
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
        // A bundle at a schema this build doesn't read is still ours to
        // replace; it is the one file an export exists to bring up to date.
        BundleFormat::Sqlite => match SqliteBundle::open(out) {
            Ok(bundle) => Some(std::format!("'{}'", bundle.manifest().name)),
            Err(BundleError::UnsupportedDatabase(schema)) => {
                Some(std::format!("(database schema {schema})"))
            }
            Err(_) => None,
        },
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

/// Removes a staged bundle and whatever the engine left beside it.
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
/// The sidecars are dropped, not moved: a database bundle was checkpointed,
/// so its `-wal` is empty, and a stale one next to `out` would otherwise be
/// replayed into the new bundle.
fn publish(staged: &Path, out: &Path) -> Result<(), BundleError> {
    for suffix in SIDECARS {
        for stale in [sidecar(out, suffix), sidecar(staged, suffix)] {
            if stale.exists() {
                std::fs::remove_file(stale)?;
            }
        }
    }

    std::fs::rename(staged, out)?;

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

pub(super) fn storage_error(error: turso::Error) -> BundleError {
    BundleError::Storage(error.to_string())
}
