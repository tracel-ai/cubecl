use alloc::string::String;
use alloc::vec::Vec;

use crate::persistence::Database;

use super::{Bundle, BundleError, BundleManifest};

/// A bundle stored as a single `SQLite` file, the format produced by
/// [`export`](super::export) on native targets.
#[derive(Debug)]
pub struct SqliteBundle {
    database: Database,
    manifest: BundleManifest,
}

impl SqliteBundle {
    /// Opens a bundle file read-only, reading and validating its manifest.
    ///
    /// A version mismatch is not an error: the bundle still installs, its
    /// entries are simply never looked up, because the cubecl version is part
    /// of every namespace. A clear warning is logged instead of silent
    /// emptiness.
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self, BundleError> {
        let path = path.as_ref();
        let database = Database::open(path, true)?;
        let manifest = BundleManifest::read(&database)?;

        if manifest.cubecl_version != env!("CARGO_PKG_VERSION") {
            log::warn!(
                "Bundle '{}' was built for cubecl {}, running {}; its entries will be ignored.",
                manifest.name,
                manifest.cubecl_version,
                env!("CARGO_PKG_VERSION"),
            );
        }

        Ok(Self { database, manifest })
    }

    /// The bundle manifest.
    pub fn manifest(&self) -> &BundleManifest {
        &self.manifest
    }

    /// The bundle database, for reporting and re-export.
    pub fn database(&self) -> &Database {
        &self.database
    }
}

impl Bundle for SqliteBundle {
    fn get(&self, namespace: &str, key: &[u8]) -> Option<Vec<u8>> {
        self.database.get(namespace, key)
    }

    fn scan(&self, namespace: &str, visit: &mut dyn FnMut(Vec<u8>, Vec<u8>)) {
        self.database.scan(namespace, visit)
    }

    fn describe(&self) -> String {
        alloc::format!(
            "bundle '{}' at {:?}",
            self.manifest.name,
            self.database.path()
        )
    }
}
