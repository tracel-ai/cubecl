//! What names a compiled artifact in a compilation store.

use crate::id::KernelId;
use cubecl_common::hash::{StableHash, StableHasher};
use cubecl_environment::environment::BuildStamp;

/// Key for an entry in the persistent compilation cache.
///
/// The [id](KernelId) covers the kernel type, its comptime arguments and its
/// launch settings, but nothing of the body: the build is what keeps an
/// artifact from outliving the code it was compiled from, since any rebuild
/// changes it.
///
/// Unknown fields refuse to decode: an environment holds an application's
/// entries beside the compilers', and one of them must never read as a key.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(deny_unknown_fields)]
pub struct KernelCacheKey {
    /// Hash of the [kernel id](KernelId): what the
    /// [work's launches](super::WorkLaunchStore) record, so an artifact is
    /// known by its kernel, whichever variant it is.
    pub id: StableHash,
    /// Hash of what else decides the artifact for that kernel: the variant a
    /// backend compiles it in, the device properties its code is specialized
    /// for. Zero when the kernel id says everything; not written then, so such
    /// a key encodes as `id` and `built_by` alone.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub variant: StableHash,
    /// The executable that compiled the artifact. Stored under the name
    /// keys were first written with, so they still decode.
    #[serde(rename = "build_id")]
    pub built_by: BuildStamp,
}

/// Whether a key's variant is the one a kernel id says everything for, which
/// is not written.
fn is_zero(variant: &StableHash) -> bool {
    *variant == 0
}

impl KernelCacheKey {
    /// The key of `id`, as the executable `built_by` compiled it.
    pub fn new(id: &KernelId, built_by: BuildStamp) -> Self {
        Self {
            id: id.stable_hash(),
            variant: 0,
            built_by,
        }
    }

    /// The key of the same kernel, compiled in `variant`.
    pub fn in_variant(self, variant: StableHash) -> Self {
        Self { variant, ..self }
    }

    /// The key of an artifact id (kernel and variant), as the executable
    /// `built_by` compiled it.
    pub fn from_artifact_id<V: core::hash::Hash>(
        id: &super::ArtifactId<V>,
        built_by: BuildStamp,
    ) -> Self {
        Self::new(&id.kernel, built_by).in_variant(StableHasher::hash_one(&id.variant))
    }

    /// The key of the artifact a compilation store's entry is of: the entry's
    /// key for an artifact, its value for an entry mapping a source to one.
    /// `None` for an entry of anything else.
    #[cfg(compilation_cache)]
    pub fn from_entry(key: &[u8], value: &[u8]) -> Option<Self> {
        let decode = |bytes: &[u8]| ciborium::from_reader::<Self, _>(bytes).ok();
        decode(key).or_else(|| decode(value))
    }
}

#[cfg(all(test, compilation_cache))]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// A key written before variants were stored, and one in the kernel's own
    /// variant, read back as the same key: the variant is written only when
    /// it says something. The stamp is written as the bare hash under
    /// `build_id`, as keys always were, so an environment's artifacts still
    /// load.
    #[test]
    fn a_key_without_a_variant_reads_as_variant_zero() {
        let key = KernelCacheKey {
            id: 1,
            variant: 0,
            built_by: BuildStamp { hash: 2 },
        };
        let written = cbor(&key);
        assert_eq!(written, cbor(&Pair { id: 1, build_id: 2 }));
        assert_eq!(KernelCacheKey::from_entry(&written, &[]), Some(key));
    }

    /// An application's entry that carries a key's fields among its own is
    /// not a key.
    #[test]
    fn an_entry_with_fields_of_its_own_is_not_a_key() {
        let entry = cbor(&Extended {
            id: 1,
            build_id: 2,
            name: "settings",
        });
        assert_eq!(KernelCacheKey::from_entry(&entry, &entry), None);
    }

    #[derive(serde::Serialize)]
    struct Pair {
        id: StableHash,
        build_id: StableHash,
    }

    #[derive(serde::Serialize)]
    struct Extended {
        id: StableHash,
        build_id: StableHash,
        name: &'static str,
    }

    fn cbor<T: serde::Serialize>(value: &T) -> Vec<u8> {
        let mut bytes = Vec::new();
        ciborium::into_writer(value, &mut bytes).unwrap();
        bytes
    }
}
