//! Which executable wrote something into an environment.

/// Identifies an executable: the hash of its build id, which any rebuild
/// changes.
///
/// A compiled kernel is stored under the stamp of the executable that
/// compiled it, since an artifact must not outlive the code it came from, and
/// a bundle carries the stamp of the executable that saved it, so a reader
/// tells its own saves from another build's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct BuildStamp {
    /// The hash of the build id.
    pub hash: u128,
}

#[cfg(not(target_os = "none"))]
impl BuildStamp {
    /// The stamp of the running executable.
    ///
    /// Where the platform exposes no build id, every executable shares one
    /// stamp: telling builds apart then falls back to the cubecl version,
    /// which every namespace carries.
    pub fn current() -> Self {
        *CURRENT
    }
}

/// Read once: the build id never changes while the executable runs.
#[cfg(not(target_os = "none"))]
static CURRENT: crate::sync::LazyLock<BuildStamp> = crate::sync::LazyLock::new(|| {
    use core::hash::Hash;

    let mut hasher = xxhash_rust::xxh3::Xxh3Default::new();
    buildid::build_id().hash(&mut hasher);
    BuildStamp {
        hash: hasher.digest128(),
    }
});

#[cfg(test)]
mod tests {
    use super::*;

    /// A stamp written into a manifest reads back as the same stamp, so a
    /// reader recognizes its own executable's saves.
    #[test]
    fn a_stamp_reads_back_from_json() {
        let stamp = BuildStamp {
            hash: u128::MAX - 7,
        };
        let json = serde_json::to_string(&stamp).unwrap();
        assert_eq!(serde_json::from_str::<BuildStamp>(&json).unwrap(), stamp);
    }

    /// The running executable keeps one stamp.
    #[test]
    fn the_current_stamp_is_stable() {
        assert_eq!(BuildStamp::current(), BuildStamp::current());
    }
}
