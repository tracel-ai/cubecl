use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::environment::BuildStamp;

/// The `meta` key the manifest is stored under.
#[cfg(native_cache)]
pub(super) const MANIFEST_KEY: &str = "manifest";

/// The manifest schema version this build reads and writes.
pub const MANIFEST_SCHEMA: u32 = 1;

/// Error opening or creating a bundle.
#[derive(Debug)]
pub enum BundleError {
    /// The bundle file couldn't be read or written.
    #[cfg(native_cache)]
    Io(std::io::Error),
    /// The environment database couldn't be opened or queried.
    #[cfg(native_cache)]
    Storage(String),
    /// The file opened but carries no manifest, so it isn't a bundle.
    NotABundle,
    /// The manifest is not valid for the expected schema.
    InvalidManifest(String),
    /// The manifest declares a schema this build doesn't understand.
    UnsupportedSchema(u32),
    /// The bundle database is at a schema this build doesn't read.
    #[cfg(native_cache)]
    UnsupportedDatabase(String),
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
            BundleError::Storage(err) => write!(f, "bundle storage error: {err}"),
            BundleError::NotABundle => write!(f, "the file carries no bundle manifest"),
            BundleError::InvalidManifest(err) => write!(f, "invalid bundle manifest: {err}"),
            BundleError::UnsupportedSchema(schema) => {
                write!(
                    f,
                    "unsupported bundle schema {schema} (this build supports {MANIFEST_SCHEMA})"
                )
            }
            #[cfg(native_cache)]
            BundleError::UnsupportedDatabase(schema) => {
                write!(
                    f,
                    "unsupported bundle database schema {schema} (this build reads {})",
                    crate::persistence::database::SCHEMA_VERSION
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
impl From<crate::persistence::sql::SqlError> for BundleError {
    fn from(err: crate::persistence::sql::SqlError) -> Self {
        Self::Storage(err.to_string())
    }
}

/// The manifest of an environment bundle, stored as a row of the bundle
/// database in [`BundleFormat::Sqlite`](super::BundleFormat::Sqlite) and as the
/// metadata blob in [`BundleFormat::Flat`](super::BundleFormat::Flat).
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
    /// The executable that exported the bundle, which tells a reader whether
    /// its own build saved it; `None` when the manifest names none.
    ///
    /// Written as the stamp's 32 hexadecimal digits: the manifest is JSON a
    /// person or a tool outside Rust reads, and a 128-bit integer loses
    /// digits in a JSON reader that parses numbers as doubles.
    #[serde(default, skip_serializing_if = "Option::is_none", with = "hex_stamp")]
    pub saved_by: Option<BuildStamp>,
}

/// [`BundleManifest::saved_by`] as hexadecimal text.
mod hex_stamp {
    use super::BuildStamp;
    use alloc::string::{String, ToString};
    use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error};

    pub fn serialize<S: Serializer>(
        stamp: &Option<BuildStamp>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        stamp.map(|stamp| stamp.to_string()).serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<BuildStamp>, D::Error> {
        Option::<String>::deserialize(deserializer)?
            .map(|hex| hex.parse().map_err(D::Error::custom))
            .transpose()
    }
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
    /// Parses and validates a serialized manifest, the JSON both formats
    /// store.
    ///
    /// This is the guard [`SqliteBundle`](super::SqliteBundle) applies at open,
    /// available to the flat format too through
    /// [`EmbeddedBundle::manifest`](super::EmbeddedBundle::manifest).
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

    /// The manifest of the database `database` opened.
    ///
    /// # Errors
    ///
    /// [`NotABundle`](BundleError::NotABundle) for an environment no export
    /// wrote, whatever [`parse`](Self::parse) refuses in the one it holds,
    /// and a storage error when the database can't be read.
    #[cfg(native_cache)]
    pub fn new(database: &crate::persistence::Database) -> Result<Self, BundleError> {
        database.with_connection(Self::read)
    }

    /// Writes this manifest into the database `database` opened, making it a
    /// bundle [`SqliteBundle`](super::SqliteBundle) reads — or replacing the
    /// manifest it had.
    ///
    /// # Errors
    ///
    /// When the database refuses the write, a read-only one among them.
    #[cfg(native_cache)]
    pub fn write_to(&self, database: &crate::persistence::Database) -> Result<(), BundleError> {
        database.with_connection(|connection| self.write(connection))
    }

    /// Reads and validates the manifest of a bundle database.
    #[cfg(native_cache)]
    pub(super) fn read(
        connection: &crate::persistence::sql::Connection,
    ) -> Result<Self, BundleError> {
        let content = crate::persistence::database::meta_get(connection, MANIFEST_KEY)
            .map_err(super::sqlite::missing_meta)?
            .ok_or(BundleError::NotABundle)?;

        Self::parse(content.as_bytes())
    }

    /// Writes the manifest into a bundle database.
    #[cfg(native_cache)]
    pub(super) fn write(
        &self,
        connection: &crate::persistence::sql::Connection,
    ) -> Result<(), BundleError> {
        let content = serde_json::to_string_pretty(self)
            .map_err(|err| BundleError::InvalidManifest(err.to_string()))?;

        crate::persistence::database::meta_set(connection, MANIFEST_KEY, &content)
            .map_err(BundleError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stamp a manifest carries is written as text, so a JSON reader
    /// that parses numbers as doubles keeps every digit, and reads back as
    /// the same stamp.
    #[test]
    fn the_saving_stamp_is_written_as_hex_text() {
        let stamp = BuildStamp {
            hash: u128::MAX - 7,
        };
        let manifest = BundleManifest {
            schema: MANIFEST_SCHEMA,
            name: "saved".into(),
            cubecl_version: "0.12.0".into(),
            created_unix_secs: None,
            environments: Vec::new(),
            saved_by: Some(stamp),
        };

        let json = serde_json::to_value(&manifest).unwrap();
        assert_eq!(json["saved_by"], serde_json::json!(stamp.to_string()));
        let read = BundleManifest::parse(json.to_string().as_bytes()).unwrap();
        assert_eq!(read.saved_by, Some(stamp));
    }
}
