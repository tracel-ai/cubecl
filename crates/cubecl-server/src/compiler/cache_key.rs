//! What names a compiled artifact in a compilation store.

use crate::id::KernelId;
use cubecl_common::hash::{StableHash, StableHasher};

/// Key for an entry in the persistent compilation cache.
///
/// The [id](KernelId) alone doesn't describe what a kernel does: it covers the kernel type, its
/// comptime arguments and its launch settings, but nothing of the body. Pairing it with a hash of
/// the expanded IR is what lets a cached artifact be invalidated when the code behind it changes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct KernelCacheKey {
    /// Hash of the [kernel id](KernelId): what the
    /// [work's launches](WorkLaunchStore) record, so an artifact is known by
    /// its kernel, whichever variant it is.
    pub id: StableHash,
    /// Hash of what else decides the artifact for that kernel: the variant a
    /// backend compiles it in, the device properties its code is specialized
    /// for. Zero when the kernel id says everything; not written then, so such
    /// a key encodes as `id` and `build_id` alone.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub variant: StableHash,
    /// Hash of the [build id](buildid::build_id).
    pub build_id: StableHash,
}

/// Whether a key's variant is the one a kernel id says everything for, which
/// is not written.
fn is_zero(variant: &StableHash) -> bool {
    *variant == 0
}

impl KernelCacheKey {
    /// Create a key from a kernel id and the current build ID.
    pub fn new(id: &KernelId, build_id: StableHash) -> Self {
        Self {
            id: id.stable_hash(),
            variant: 0,
            build_id,
        }
    }

    /// The key of the same kernel, compiled in `variant`.
    pub fn in_variant(self, variant: StableHash) -> Self {
        Self { variant, ..self }
    }

    /// Create a key from an artifact id (kernel and variant) and the current build ID.
    pub fn from_artifact_id<V: core::hash::Hash>(
        id: &super::ArtifactId<V>,
        build_id: StableHash,
    ) -> Self {
        Self::new(&id.kernel, build_id).in_variant(StableHasher::hash_one(&id.variant))
    }

    /// The artifact a compilation store's entry is of: its key for an
    /// artifact, its value for an entry mapping a source to one. `None` for
    /// an entry of anything else.
    #[cfg(compilation_cache)]
    pub fn of_entry(key: &[u8], value: &[u8]) -> Option<Self> {
        let decode = |bytes: &[u8]| ciborium::from_reader::<Self, _>(bytes).ok();
        decode(key).or_else(|| decode(value))
    }
}
