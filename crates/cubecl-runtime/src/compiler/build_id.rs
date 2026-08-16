use cubecl_common::hash::{StableHash, StableHasher};

/// Fork of `buildid`'s Windows logic to fix incremental builds. If upstream gets a fix, this can
/// be removed. Linux should work correctly because it uses deterministic build IDs and doesn't
/// keep them on incremental builds from all the info I could find.
#[cfg(target_os = "windows")]
mod windows;

/// Platform-specific build identifier, changes on rebuild
pub type BuildId = Option<&'static [u8]>;

/// Read the build ID for the currently loaded binary
pub fn build_id() -> BuildId {
    cfg_select! {
        windows => windows::build_id(),
        _ => buildid::build_id()
    }
}

/// Pre-hashed build ID
pub fn build_id_hash() -> StableHash {
    StableHasher::hash_one(&build_id())
}
