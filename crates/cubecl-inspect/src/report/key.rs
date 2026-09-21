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

/// One autotune table: the namespace cubecl stores a tuner's results in,
/// `autotune/<cubecl version>/<device>/<tuner>`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct AutotuneTable {
    /// The tuner, named by the code that declared it.
    pub tuner: String,
    /// The device the results were measured on.
    pub device: String,
    /// The cubecl version that wrote them.
    pub version: String,
}

impl AutotuneTable {
    /// The namespace's root.
    pub const ROOT: &str = "autotune";

    /// The namespace the table is stored under.
    pub fn namespace(&self) -> String {
        format!(
            "{}/{}/{}/{}",
            Self::ROOT,
            self.version,
            self.device,
            self.tuner
        )
    }

    /// The table a namespace names, or `None` when it is not an autotune one.
    pub fn parse(namespace: &str) -> Option<Self> {
        let mut segments = namespace.splitn(4, '/');
        if segments.next()? != Self::ROOT {
            return None;
        }
        Some(Self {
            version: segments.next()?.to_string(),
            device: segments.next()?.to_string(),
            tuner: segments.next()?.to_string(),
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

    #[test]
    fn a_table_parses_from_its_namespace() {
        assert_eq!(
            AutotuneTable::parse("autotune/0.11.0/device-32-0-hip/matmul-tune-gemm"),
            Some(AutotuneTable {
                tuner: "matmul-tune-gemm".to_string(),
                device: "device-32-0-hip".to_string(),
                version: "0.11.0".to_string(),
            })
        );
        let namespace = "autotune/0.11.0/device-32-0-hip/matmul-tune-gemm";
        let table = AutotuneTable::parse(namespace).expect("parses");
        assert_eq!(table.namespace(), namespace);
        assert_eq!(AutotuneTable::parse("hip/0.11.0/gfx1151"), None);
    }
}
