use alloc::string::String;
use alloc::vec::Vec;

/// A read-only source of pre-seeded cache data.
///
/// Lookups are keyed by the logical store name the persistence layer already
/// computes (`<name>/<version>/<segments>`) and the serialized key bytes, so a
/// bundle addresses entries exactly like the local cache does.
///
/// All methods degrade silently: a miss on any failure.
pub trait SeedSource: Send + Sync + core::fmt::Debug {
    /// The value stored under `key` in `store`.
    fn get(&self, store: &str, key: &[u8]) -> Option<Vec<u8>>;

    /// Visits every entry of `store`.
    fn scan(&self, store: &str, visit: &mut dyn FnMut(Vec<u8>, Vec<u8>));

    /// Human-readable origin for log messages.
    fn describe(&self) -> String;
}

/// A bundle loaded from a file on the file system.
#[cfg(feature = "cache")]
#[derive(Debug)]
pub struct Bundle {
    database: crate::persistence::Database,
    manifest: super::BundleManifest,
}

#[cfg(feature = "cache")]
impl Bundle {
    /// Opens a bundle file read-only, reading and validating its manifest.
    ///
    /// A version mismatch is not an error: the bundle still installs, its
    /// entries are simply never looked up, because the cubecl version is part
    /// of every store name. A clear warning is logged instead of silent
    /// emptiness.
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self, super::BundleError> {
        let path = path.as_ref();
        let database = crate::persistence::Database::open(path, true)?;
        let manifest = super::BundleManifest::read(&database)?;

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
    pub fn manifest(&self) -> &super::BundleManifest {
        &self.manifest
    }

    /// The bundle database, for reporting and re-export.
    pub fn database(&self) -> &crate::persistence::Database {
        &self.database
    }
}

#[cfg(feature = "cache")]
impl SeedSource for Bundle {
    fn get(&self, store: &str, key: &[u8]) -> Option<Vec<u8>> {
        self.database.get(store, key)
    }

    fn scan(&self, store: &str, visit: &mut dyn FnMut(Vec<u8>, Vec<u8>)) {
        self.database.scan(store, visit)
    }

    fn describe(&self) -> String {
        alloc::format!(
            "bundle '{}' at {:?}",
            self.manifest.name,
            self.database.path()
        )
    }
}
