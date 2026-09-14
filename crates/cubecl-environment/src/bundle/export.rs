use std::path::{Path, PathBuf};
use std::string::{String, ToString};
use std::vec::Vec;

use crate::bytes::Bytes;

use super::flat;
use super::{BundleError, BundleManifest, EnvironmentInfo, MANIFEST_SCHEMA, flat_bundle_version};

const SELECT: &str = "SELECT namespace, key, value FROM cache_entries";
/// A plain prefix match on whole segments, avoiding LIKE's wildcards.
const NAMESPACE_PREFIX: &str = "namespace = ?1 OR substr(namespace, 1, length(?1) + 1) = ?1 || '/'";

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
}

/// Copies entries from one or more cache roots into a bundle file.
///
/// Merging several roots deduplicates by `(namespace, key)` rather than
/// concatenating bytes, and restricting the export to a few namespaces is a
/// filter.
///
/// The typical workflow: run the application once so autotune and the
/// compilation caches are warm, then export the cache root.
///
/// The bundle is built next to `out` and renamed onto it once complete, so a
/// failed export leaves the previous bundle, or no file at all, rather than a
/// truncated one.
pub async fn export<R: AsRef<Path>, O: AsRef<Path>>(
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

    // Nothing writes to `out` until the bundle is complete, so an interrupted
    // export can't leave a file that later exports refuse to overwrite.
    let staged = staging_path(out);
    discard(&staged);

    let mut entries = flat::Entries::new();
    let namespaces = filters(&options.namespaces);
    let result = async {
        for source in &sources {
            read_entries(source, namespaces, &mut entries).await?;
        }
        flat::write(&staged, &entries, &manifest)
    }
    .await;

    if let Err(err) = result {
        discard(&staged);
        return Err(err);
    }

    publish(&staged, out)?;

    if entries.is_empty() {
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
/// filter at all.
fn filters(namespaces: &[String]) -> Option<&[String]> {
    let unrestricted = namespaces.is_empty() || namespaces.iter().any(String::is_empty);
    (!unrestricted).then_some(namespaces)
}

/// Collects the requested namespaces of the database at `source` into
/// `entries`. The first root to export a key wins on collision.
async fn read_entries(
    source: &Path,
    namespaces: Option<&[String]>,
    entries: &mut flat::Entries,
) -> Result<(), BundleError> {
    let location = source.to_str().ok_or_else(|| {
        BundleError::Storage(std::format!("cache path {source:?} is not valid UTF-8"))
    })?;
    let database = turso::Builder::new_local(location)
        .experimental_multiprocess_wal(true)
        .build()
        .await
        .map_err(storage_error)?;
    let connection = database.connect().map_err(storage_error)?;

    match namespaces {
        None => {
            collect(
                connection.query(SELECT, ()).await.map_err(storage_error)?,
                entries,
            )
            .await
        }
        Some(namespaces) => {
            let query = std::format!("{SELECT} WHERE {NAMESPACE_PREFIX}");
            for namespace in namespaces {
                let rows = connection
                    .query(&query, (namespace.as_str(),))
                    .await
                    .map_err(storage_error)?;
                collect(rows, entries).await?;
            }
            Ok(())
        }
    }
}

async fn collect(mut rows: turso::Rows, entries: &mut flat::Entries) -> Result<(), BundleError> {
    while let Some(row) = rows.next().await.map_err(storage_error)? {
        let namespace: String = row.get(0).map_err(storage_error)?;
        let key: Vec<u8> = row.get(1).map_err(storage_error)?;
        let value: Vec<u8> = row.get(2).map_err(storage_error)?;
        entries
            .entry((namespace, key))
            .or_insert_with(|| Bytes::from_bytes_vec(value));
    }
    Ok(())
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

    // A file is ours when it carries the flat header; anything else is left
    // alone rather than overwritten. The manifest lives inside the blob, so
    // the header identifies the file; reading all of it just to name it isn't
    // worth it.
    match flat_header(out) {
        Some(version) => log::info!("Replacing the existing flat v{version} bundle at {out:?}"),
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

/// Removes a staged bundle.
fn discard(staged: &Path) {
    if staged.exists()
        && let Err(err) = std::fs::remove_file(staged)
    {
        log::warn!("Bundle export: can't remove {staged:?}: {err}");
    }
}

/// Moves the finished bundle onto `out`, replacing what was there.
fn publish(staged: &Path, out: &Path) -> Result<(), BundleError> {
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

fn storage_error(error: turso::Error) -> BundleError {
    BundleError::Storage(error.to_string())
}
