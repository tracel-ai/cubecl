use std::string::{String, ToString};
use std::vec::Vec;

use crate::persistence::Database;

/// The `meta` key the manifest is stored under.
const MANIFEST_KEY: &str = "manifest";

/// The manifest schema version this build reads and writes.
pub const MANIFEST_SCHEMA: u32 = 1;

/// Error opening or creating a bundle.
#[derive(Debug)]
pub enum BundleError {
    /// The bundle file couldn't be read or written.
    Io(std::io::Error),
    /// The database couldn't be opened or queried.
    Database(rusqlite::Error),
    /// The file opened but carries no manifest, so it isn't a bundle.
    NotABundle,
    /// The manifest is not valid for the expected schema.
    InvalidManifest(String),
    /// The manifest declares a schema this build doesn't understand.
    UnsupportedSchema(u32),
}

impl core::fmt::Display for BundleError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BundleError::Io(err) => write!(f, "bundle io error: {err}"),
            BundleError::Database(err) => write!(f, "bundle database error: {err}"),
            BundleError::NotABundle => write!(f, "the file carries no bundle manifest"),
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

impl From<rusqlite::Error> for BundleError {
    fn from(err: rusqlite::Error) -> Self {
        Self::Database(err)
    }
}

/// The manifest of an environment bundle, stored as a row of the bundle
/// database.
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
    /// Reads and validates the manifest of a bundle database.
    pub fn read(database: &Database) -> Result<Self, BundleError> {
        let content = read_meta(database, MANIFEST_KEY)?.ok_or(BundleError::NotABundle)?;

        let manifest: Self = serde_json::from_str(&content)
            .map_err(|err| BundleError::InvalidManifest(err.to_string()))?;

        if manifest.schema != MANIFEST_SCHEMA {
            return Err(BundleError::UnsupportedSchema(manifest.schema));
        }

        Ok(manifest)
    }

    /// Writes the manifest into a bundle database.
    pub fn write(&self, database: &Database) -> Result<(), BundleError> {
        let content = serde_json::to_string_pretty(self)
            .map_err(|err| BundleError::InvalidManifest(err.to_string()))?;

        database.with_connection(|conn| {
            conn.execute(
                "INSERT INTO meta (k, v) VALUES (?1, ?2) \
                 ON CONFLICT(k) DO UPDATE SET v = excluded.v",
                rusqlite::params![MANIFEST_KEY, content],
            )
        })?;

        Ok(())
    }
}

fn read_meta(database: &Database, key: &str) -> Result<Option<String>, BundleError> {
    use rusqlite::OptionalExtension;

    let content = database.with_connection(|conn| {
        conn.query_row(
            "SELECT v FROM meta WHERE k = ?1",
            rusqlite::params![key],
            |row| row.get::<_, String>(0),
        )
        .optional()
    });

    match content {
        Ok(content) => Ok(content),
        // A file that isn't a cubecl database has no `meta` table at all.
        Err(rusqlite::Error::SqliteFailure(..)) | Err(rusqlite::Error::SqlInputError { .. }) => {
            Err(BundleError::NotABundle)
        }
        Err(err) => Err(BundleError::Database(err)),
    }
}
