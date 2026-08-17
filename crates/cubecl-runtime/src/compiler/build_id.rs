use cubecl_common::hash::{StableHash, StableHasher};

/// Platform-specific build identifier, changes on rebuild
pub type BuildId = Option<&'static [u8]>;

/// Pre-hashed build ID
pub fn build_id_hash() -> StableHash {
    StableHasher::hash_one(&buildid::build_id())
}
