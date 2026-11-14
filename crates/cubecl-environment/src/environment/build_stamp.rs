//! Which executable wrote something into an environment.

/// Identifies an executable: the hash of its build id, which any rebuild
/// changes.
///
/// A compiled kernel is stored under the stamp of the executable that
/// compiled it, since an artifact must not outlive the code it came from, and
/// a bundle carries the stamp of the executable that saved it, so a reader
/// tells its own saves from another build's.
///
/// Equal stamps mean the same build only where the platform exposes a build
/// id. Where it exposes none, and on an embedded target, every executable
/// shares one stamp: two builds of one cubecl version then load each other's
/// kernels and read each other's saves as their own, and only the cubecl
/// version, which every namespace carries, still tells builds apart.
///
/// Serialized as the hash itself, the encoding the compilation stores key an
/// artifact by; [`Display`](core::fmt::Display) and
/// [`FromStr`](core::str::FromStr) spell it as 32 hexadecimal digits, for a
/// text format.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(transparent)]
pub struct BuildStamp {
    /// The hash of the build id.
    pub hash: u128,
}

impl BuildStamp {
    /// The stamp of the running executable.
    pub fn current() -> Self {
        current()
    }
}

impl core::fmt::Display for BuildStamp {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:032x}", self.hash)
    }
}

impl core::str::FromStr for BuildStamp {
    type Err = core::num::ParseIntError;

    fn from_str(hex: &str) -> Result<Self, Self::Err> {
        u128::from_str_radix(hex, 16).map(|hash| Self { hash })
    }
}

#[cfg(not(target_os = "none"))]
fn current() -> BuildStamp {
    // Read once: the build id never changes while the executable runs.
    static CURRENT: crate::sync::LazyLock<BuildStamp> = crate::sync::LazyLock::new(|| {
        use core::hash::Hash;

        let mut hasher = xxhash_rust::xxh3::Xxh3Default::new();
        buildid::build_id().hash(&mut hasher);
        BuildStamp {
            hash: hasher.digest128(),
        }
    });
    *CURRENT
}

/// An embedded target has no executable to read a build id from, so every
/// one shares this stamp.
#[cfg(target_os = "none")]
fn current() -> BuildStamp {
    BuildStamp { hash: 0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    /// A stamp written into JSON reads back as the same stamp, so a reader
    /// recognizes its own executable's saves.
    #[test]
    fn a_stamp_reads_back_from_json() {
        let stamp = BuildStamp {
            hash: u128::MAX - 7,
        };
        let json = serde_json::to_string(&stamp).unwrap();
        assert_eq!(serde_json::from_str::<BuildStamp>(&json).unwrap(), stamp);
    }

    /// A stamp spelled in hexadecimal reads back as the same stamp, its
    /// leading zeros included.
    #[test]
    fn a_stamp_reads_back_from_its_hex_spelling() {
        let stamp = BuildStamp { hash: 0xff };
        let hex = stamp.to_string();
        assert_eq!(hex.len(), 32);
        assert_eq!(hex.parse::<BuildStamp>().unwrap(), stamp);
    }
}
