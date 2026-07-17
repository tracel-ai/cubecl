use std::path::Path;
use std::string::{String, ToString};
use std::vec::Vec;

/// File name of the bundle manifest.
pub const MANIFEST_FILE_NAME: &str = "bundle.toml";

/// Directory holding the cache tree inside a bundle.
pub const STORE_DIR_NAME: &str = "store";

/// The manifest schema version this build reads and writes.
pub const MANIFEST_SCHEMA: u32 = 1;

/// Error opening or creating a bundle.
#[derive(Debug)]
pub enum BundleError {
    /// The manifest file couldn't be read or written.
    Io(std::io::Error),
    /// The manifest is not valid TOML for the expected schema.
    InvalidManifest(String),
    /// The manifest declares a schema this build doesn't understand.
    UnsupportedSchema(u32),
}

impl core::fmt::Display for BundleError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BundleError::Io(err) => write!(f, "bundle io error: {err}"),
            BundleError::InvalidManifest(err) => write!(f, "invalid bundle manifest: {err}"),
            BundleError::UnsupportedSchema(schema) => {
                write!(
                    f,
                    "unsupported bundle schema {schema} (this build supports {MANIFEST_SCHEMA})"
                )
            }
        }
    }
}

impl core::error::Error for BundleError {}

impl From<std::io::Error> for BundleError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

/// The `bundle.toml` manifest of an environment bundle.
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
    /// Reads and validates a manifest file.
    pub fn read(path: &Path) -> Result<Self, BundleError> {
        let content = std::fs::read_to_string(path)?;
        let manifest: Self = toml::from_str(&content)
            .map_err(|err| BundleError::InvalidManifest(err.to_string()))?;

        if manifest.schema != MANIFEST_SCHEMA {
            return Err(BundleError::UnsupportedSchema(manifest.schema));
        }

        Ok(manifest)
    }

    /// Serializes the manifest to a file.
    pub fn write(&self, path: &Path) -> Result<(), BundleError> {
        let content = toml::to_string_pretty(self)
            .map_err(|err| BundleError::InvalidManifest(err.to_string()))?;
        std::fs::write(path, content)?;
        Ok(())
    }
}
