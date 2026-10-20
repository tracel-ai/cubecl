use serde::{Serialize, Serializer};
use std::fmt;
use std::str::FromStr;

/// A stable handle on one stored autotune key: a hash of its namespace and its
/// key bytes, so the same key carries the same id in every file and every run,
/// which is what lets a command line name it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct KeyId(u32);

impl KeyId {
    /// FNV-1a: stable across builds and platforms, which `std`'s hasher is
    /// explicitly not, and an id only has to tell a file's keys apart.
    pub fn new(namespace: &str, key: &[u8]) -> Self {
        const OFFSET: u32 = 0x811c_9dc5;
        const PRIME: u32 = 0x0100_0193;

        let hash = namespace
            .as_bytes()
            .iter()
            .chain([0u8].iter())
            .chain(key)
            .fold(OFFSET, |hash, byte| {
                (hash ^ u32::from(*byte)).wrapping_mul(PRIME)
            });
        Self(hash)
    }
}

impl fmt::Display for KeyId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:08x}", self.0)
    }
}

impl FromStr for KeyId {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        u32::from_str_radix(text, 16)
            .map(Self)
            .map_err(|_| format!("`{text}` is not a key id (eight hex digits)"))
    }
}

/// The id as it is printed, so JSON and text name a key the same way.
impl Serialize for KeyId {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

/// A decoded key's canonical rendering: what matches one key across the
/// records and the files that carry it, since `ciborium::Value` is neither
/// `Ord` nor `Hash`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct CanonicalKey(String);

impl CanonicalKey {
    pub(crate) fn new(key: &ciborium::Value) -> Self {
        Self(format!("{key:?}"))
    }
}

/// One autotune table: the namespace cubecl stores a tuner's results in,
/// as the runtime's `TuneTableName` writes it,
/// `autotune/<cubecl version>/<fingerprint>/<device>/<tuner>`, or as cubecl
/// wrote it before it keyed tables on the fingerprint,
/// `autotune/<cubecl version>/<device>/<tuner>`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct AutotuneTable {
    /// The tuner, named by the code that declared it.
    pub tuner: String,
    /// The device the results were measured on.
    pub device: String,
    /// The adapter and API the device resolved to; `None` for a table
    /// written before cubecl recorded it.
    pub fingerprint: Option<String>,
    /// The cubecl version that wrote them.
    pub version: String,
}

impl AutotuneTable {
    /// The namespace's root.
    pub const ROOT: &str = cubecl_runtime::tune::ROOT;

    /// The namespace the table is stored under.
    pub fn namespace(&self) -> String {
        let Self {
            tuner,
            device,
            fingerprint,
            version,
        } = self;
        match fingerprint {
            Some(fingerprint) => format!("{}/{version}/{fingerprint}/{device}/{tuner}", Self::ROOT),
            None => format!("{}/{version}/{device}/{tuner}", Self::ROOT),
        }
    }

    /// The table a namespace names, or `None` when it is not an autotune one.
    ///
    /// The version is read from the front, the tuner and the device from the
    /// back, and the fingerprint is what lies between: a fingerprint built
    /// from an adapter's name may hold a `/`, a tuner's name and a device id
    /// never do.
    pub fn new(namespace: &str) -> Option<Self> {
        let rest = namespace.strip_prefix(Self::ROOT)?.strip_prefix('/')?;
        let (version, rest) = rest.split_once('/')?;
        let (rest, tuner) = rest.rsplit_once('/')?;
        let (fingerprint, device) = match rest.rsplit_once('/') {
            Some((fingerprint, device)) => (Some(fingerprint.to_string()), device),
            None => (None, rest),
        };
        Some(Self {
            tuner: tuner.to_string(),
            device: device.to_string(),
            fingerprint,
            version: version.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_id_reads_back_from_its_rendering() {
        let id = KeyId::new("autotune/0.11.0/hip-0/gemm", b"key");
        assert_eq!(id.to_string().parse::<KeyId>(), Ok(id));
    }

    /// The namespace is part of the id: one key tuned by two tuners is two
    /// keys.
    #[test]
    fn the_namespace_is_part_of_the_id() {
        assert_ne!(KeyId::new("a", b"key"), KeyId::new("b", b"key"));
        assert_ne!(KeyId::new("ab", b"c"), KeyId::new("a", b"bc"));
    }

    /// What the runtime writes is what the reader parses: the two sides of
    /// one layout, in two crates.
    #[test]
    fn a_table_parses_from_the_namespace_the_runtime_writes() {
        let namespace = cubecl_runtime::tune::TuneTableName {
            fingerprint: "NVIDIA GeForce RTX 3080/PCIe/SSE2",
            device_id: "DeviceId(type=0, index=0)",
            tuner: "matmul-tune-gemm",
        }
        .namespace();
        let table = AutotuneTable::new(namespace.as_str()).expect("parses");
        assert_eq!(table.tuner, "matmul-tune-gemm");
        assert_eq!(table.device, "DeviceId(type=0, index=0)");
        assert_eq!(
            table.fingerprint.as_deref(),
            Some("NVIDIA GeForce RTX 3080/PCIe/SSE2")
        );
        assert_eq!(table.namespace(), namespace.as_str());
    }

    /// A table written before cubecl keyed tables on the fingerprint still
    /// reads, without one.
    #[test]
    fn a_table_without_a_fingerprint_still_parses() {
        let namespace = "autotune/0.11.0/device-32-0-hip/matmul-tune-gemm";
        assert_eq!(
            AutotuneTable::new(namespace),
            Some(AutotuneTable {
                tuner: "matmul-tune-gemm".to_string(),
                device: "device-32-0-hip".to_string(),
                fingerprint: None,
                version: "0.11.0".to_string(),
            })
        );
        let table = AutotuneTable::new(namespace).expect("parses");
        assert_eq!(table.namespace(), namespace);
        assert_eq!(AutotuneTable::new("hip/0.11.0/gfx1151"), None);
        assert_eq!(AutotuneTable::new("autotune/0.11.0/gemm"), None);
    }
}
