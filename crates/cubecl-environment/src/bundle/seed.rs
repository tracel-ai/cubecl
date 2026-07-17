use alloc::borrow::Cow;
use alloc::string::String;

/// A read-only source of pre-seeded cache data.
///
/// Lookups are keyed by the store-relative path the persistence layer already
/// computes (`<name>/<version>/<segments>.json.log`, `/`-separated), so
/// bundles mirror the cache root layout with no rewriting.
///
/// All methods degrade silently: `None` on any failure.
pub trait SeedSource: Send + Sync + core::fmt::Debug {
    /// Raw JSON-lines bytes for the store at the given relative path.
    fn kv_bytes(&self, rel: &str) -> Option<Cow<'_, [u8]>>;

    /// Raw bytes of a chunk file referenced by a bundled table of contents.
    fn chunk_bytes(&self, rel: &str) -> Option<Cow<'_, [u8]>> {
        let _ = rel;
        None
    }

    /// Human-readable origin for log messages.
    fn describe(&self) -> String;
}

/// A seed source embedded in the binary, typically via `include_bytes!`.
///
/// Entries are `(relative path, bytes)` pairs using the same relative paths as
/// the on-disk bundle `store/` tree. This is the bundle form usable on wasm
/// and no-std targets, where there is no file system to load a bundle from.
#[derive(Debug, Clone, Copy)]
pub struct StaticSeed {
    files: &'static [(&'static str, &'static [u8])],
}

impl StaticSeed {
    /// Creates a static seed from `(relative path, bytes)` pairs.
    pub const fn new(files: &'static [(&'static str, &'static [u8])]) -> Self {
        Self { files }
    }
}

impl SeedSource for StaticSeed {
    fn kv_bytes(&self, rel: &str) -> Option<Cow<'_, [u8]>> {
        self.files
            .iter()
            .find(|(path, _)| *path == rel)
            .map(|(_, bytes)| Cow::Borrowed(*bytes))
    }

    fn chunk_bytes(&self, rel: &str) -> Option<Cow<'_, [u8]>> {
        self.kv_bytes(rel)
    }

    fn describe(&self) -> String {
        String::from("static bundle (embedded)")
    }
}

/// A bundle loaded from a directory on the file system.
#[cfg(feature = "cache")]
#[derive(Debug)]
pub struct Bundle {
    root: std::path::PathBuf,
    manifest: super::BundleManifest,
}

#[cfg(feature = "cache")]
impl Bundle {
    /// Opens a bundle directory, reading and validating its `bundle.toml`.
    ///
    /// A version mismatch is not an error: the bundle still installs, its
    /// entries are simply never looked up. A clear warning is logged instead
    /// of silent emptiness.
    pub fn open<P: Into<std::path::PathBuf>>(path: P) -> Result<Self, super::BundleError> {
        let root = path.into();
        let manifest = super::BundleManifest::read(&root.join(super::MANIFEST_FILE_NAME))?;

        if manifest.cubecl_version != env!("CARGO_PKG_VERSION") {
            log::warn!(
                "Bundle '{}' was built for cubecl {}, running {}; its entries will be ignored.",
                manifest.name,
                manifest.cubecl_version,
                env!("CARGO_PKG_VERSION"),
            );
        }

        Ok(Self { root, manifest })
    }

    /// The bundle manifest.
    pub fn manifest(&self) -> &super::BundleManifest {
        &self.manifest
    }

    fn store_file(&self, rel: &str) -> Option<Cow<'_, [u8]>> {
        // `rel` is always produced by the persistence layer (never user
        // input), and bundle paths are sanitized at creation with the same
        // rules, so a plain join is safe.
        let path = self.root.join(super::STORE_DIR_NAME).join(rel);
        match std::fs::read(&path) {
            Ok(bytes) => Some(Cow::Owned(bytes)),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => None,
            Err(err) => {
                log::warn!(
                    "Bundle '{}': can't read {path:?}: {err}",
                    self.manifest.name
                );
                None
            }
        }
    }
}

#[cfg(feature = "cache")]
impl SeedSource for Bundle {
    fn kv_bytes(&self, rel: &str) -> Option<Cow<'_, [u8]>> {
        self.store_file(rel)
    }

    fn chunk_bytes(&self, rel: &str) -> Option<Cow<'_, [u8]>> {
        self.store_file(rel)
    }

    fn describe(&self) -> String {
        alloc::format!("bundle '{}' at {:?}", self.manifest.name, self.root)
    }
}
