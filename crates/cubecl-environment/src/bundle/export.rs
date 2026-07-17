use std::path::Path;
use std::string::{String, ToString};
use std::vec::Vec;

use super::{
    BundleError, BundleManifest, EnvironmentInfo, MANIFEST_FILE_NAME, MANIFEST_SCHEMA,
    STORE_DIR_NAME,
};

/// The cache store names exported by default: every name `CubeCL` itself
/// writes under a cache root.
///
/// An explicit list is required because the default cache root is the project
/// `target` directory, which also holds build artifacts that must never end
/// up in a bundle.
pub const DEFAULT_STORE_NAMES: &[&str] = &[
    "autotune",
    "throughput",
    "cubecl",
    "cuda",
    "hip",
    "metal",
    "vulkan",
];

/// Options for [`export`].
#[derive(Debug, Clone, Default)]
pub struct ExportOptions {
    /// Human-chosen bundle name, e.g. "H100 Linux".
    pub name: String,
    /// The environments the bundle was captured on. `os` and `arch` are
    /// auto-filled from the build target when left empty.
    pub environments: Vec<EnvironmentInfo>,
    /// The cache store names (top-level directories under each cache root) to
    /// export. Empty means [`DEFAULT_STORE_NAMES`].
    pub store_names: Vec<String>,
}

/// Snapshots one or more cache roots into a bundle directory.
///
/// Copies the cache store trees (see [`ExportOptions::store_names`]) of every
/// root into `<out>/store/` (merging them; store names like `autotune`,
/// `cuda` or `vulkan` never collide) and writes the `bundle.toml` manifest.
/// Lock sidecar files are skipped. Because cache files are append-only, a
/// snapshot taken while another process writes is at worst truncated on its
/// last line, which the parsers already tolerate.
///
/// The typical workflow: run the application once so autotune and the
/// compilation caches are warm, then export the cache root.
pub fn export<R: AsRef<Path>, O: AsRef<Path>>(
    cache_roots: &[R],
    out: O,
    options: &ExportOptions,
) -> Result<BundleManifest, BundleError> {
    let out = out.as_ref();
    let store = out.join(STORE_DIR_NAME);
    std::fs::create_dir_all(&store)?;

    let names: Vec<&str> = if options.store_names.is_empty() {
        DEFAULT_STORE_NAMES.to_vec()
    } else {
        options.store_names.iter().map(String::as_str).collect()
    };

    for root in cache_roots {
        let root = root.as_ref();
        if !root.exists() {
            log::warn!("Bundle export: cache root {root:?} does not exist, skipping.");
            continue;
        }
        for name in &names {
            let from = root.join(name);
            if from.exists() {
                let to = store.join(name);
                std::fs::create_dir_all(&to)?;
                copy_tree(&from, &to)?;
            }
        }
    }

    let mut environments = options.environments.clone();
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

    let manifest = BundleManifest {
        schema: MANIFEST_SCHEMA,
        name: options.name.clone(),
        cubecl_version: env!("CARGO_PKG_VERSION").to_string(),
        created_unix_secs: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()
            .map(|elapsed| elapsed.as_secs()),
        environments,
    };
    manifest.write(&out.join(MANIFEST_FILE_NAME))?;

    Ok(manifest)
}

fn copy_tree(from: &Path, to: &Path) -> Result<(), BundleError> {
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let path = entry.path();
        let target = to.join(entry.file_name());

        if entry.file_type()?.is_dir() {
            std::fs::create_dir_all(&target)?;
            copy_tree(&path, &target)?;
        } else {
            // Lock sidecars are transient state, never bundle them.
            if path
                .extension()
                .is_some_and(|extension| extension == "lock")
            {
                continue;
            }
            std::fs::copy(&path, &target)?;
        }
    }

    Ok(())
}
