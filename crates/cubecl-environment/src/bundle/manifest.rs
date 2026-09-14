use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// The manifest schema version this build reads and writes.
pub const MANIFEST_SCHEMA: u32 = 1;

/// Error opening or creating a bundle.
#[derive(Debug)]
pub enum BundleError {
    /// The bundle file couldn't be read or written.
    #[cfg(native_cache)]
    Io(std::io::Error),
    /// The database couldn't be opened or queried.
    #[cfg(native_cache)]
    Database(rusqlite::Error),
    /// The file opened but carries no manifest, so it isn't a bundle.
    NotABundle,
    /// The manifest is not valid for the expected schema.
    InvalidManifest(String),
    /// The manifest declares a schema this build doesn't understand.
    UnsupportedSchema(u32),
    /// The bundle exceeds what the format can address.
    TooLarge,
    /// The blob isn't a readable flat bundle.
    Flat(super::EmbeddedBundleError),
}

impl core::fmt::Display for BundleError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            #[cfg(native_cache)]
            BundleError::Io(err) => write!(f, "bundle io error: {err}"),
            #[cfg(native_cache)]
            BundleError::Database(err) => write!(f, "bundle database error: {err}"),
            BundleError::NotABundle => write!(f, "the file carries no bundle manifest"),
            BundleError::InvalidManifest(err) => write!(f, "invalid bundle manifest: {err}"),
            BundleError::UnsupportedSchema(schema) => {
                write!(
                    f,
                    "unsupported bundle schema {schema} (this build supports {MANIFEST_SCHEMA})"
                )
            }
            BundleError::TooLarge => write!(
                f,
                "the flat bundle format addresses at most {} bytes; export fewer namespaces",
                u32::MAX
            ),
            // Already a complete sentence, and prefixing it would read as two
            // diagnoses of the same failure.
            BundleError::Flat(err) => write!(f, "{err}"),
        }
    }
}

impl core::error::Error for BundleError {}

impl From<super::EmbeddedBundleError> for BundleError {
    fn from(err: super::EmbeddedBundleError) -> Self {
        Self::Flat(err)
    }
}

#[cfg(native_cache)]
impl From<std::io::Error> for BundleError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

#[cfg(native_cache)]
impl From<rusqlite::Error> for BundleError {
    fn from(err: rusqlite::Error) -> Self {
        Self::Database(err)
    }
}

/// The manifest of an environment bundle, stored as the metadata blob of the
/// flat layout.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BundleManifest {
    /// Manifest schema version.
    pub schema: u32,
    /// Human-chosen bundle name, e.g. "H100 Linux".
    pub name: String,
    /// The cubecl version the bundle was exported with. Entries are only
    /// visible to the same version, exactly like local caches.
    pub cubecl_version: String,
    /// Creation time as seconds since the unix epoch, informational.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_unix_secs: Option<u64>,
    /// The environments the bundle was captured on, informational in v1.
    #[serde(default, rename = "environments")]
    pub environments: Vec<EnvironmentInfo>,
}

/// Description of one environment a bundle was captured on. Informational:
/// correctness never depends on these fields.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct EnvironmentInfo {
    /// Short machine-friendly label, e.g. "h100-linux".
    #[serde(default)]
    pub label: String,
    /// Operating system, e.g. "linux".
    #[serde(default)]
    pub os: String,
    /// CPU architecture, e.g. `x86_64`.
    #[serde(default)]
    pub arch: String,
    /// Free-form device fingerprints, e.g. `cuda-0: NVIDIA H100 PCIe (sm_90)`.
    #[serde(default)]
    pub devices: Vec<String>,
}

impl BundleManifest {
    /// Parses and validates a serialized manifest, the JSON a bundle stores
    /// as its metadata blob.
    ///
    /// This is the guard
    /// [`EmbeddedBundle::manifest`](super::EmbeddedBundle::manifest) applies
    /// at open.
    pub fn parse(content: &[u8]) -> Result<Self, BundleError> {
        if content.is_empty() {
            return Err(BundleError::NotABundle);
        }

        let manifest: Self = serde_json::from_slice(content)
            .map_err(|err| BundleError::InvalidManifest(err.to_string()))?;

        if manifest.schema != MANIFEST_SCHEMA {
            return Err(BundleError::UnsupportedSchema(manifest.schema));
        }

        Ok(manifest)
    }

    /// Warns when the bundle was built for another cubecl version.
    ///
    /// Not an error: the bundle still installs, its entries are simply never
    /// looked up, because the cubecl version is part of every namespace. A
    /// clear warning beats silent emptiness.
    pub fn warn_on_version_mismatch(&self) {
        if self.cubecl_version != env!("CARGO_PKG_VERSION") {
            log::warn!(
                "Bundle '{}' was built for cubecl {}, running {}; its entries will be ignored.",
                self.name,
                self.cubecl_version,
                env!("CARGO_PKG_VERSION"),
            );
        }
    }
}
