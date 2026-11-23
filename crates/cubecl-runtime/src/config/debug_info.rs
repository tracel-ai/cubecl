use super::compilation::PROFILE_DEBUG_INFO;
use cubecl_ir::settings::DebugInfo;

/// The debug data of a kernel that asks for `requested`, with the global configuration: the same
/// level as [`CompilationConfig::resolve_debug_info`]. It reads the configuration once, so a kernel
/// id can call it at each launch.
pub fn effective_debug_info(requested: DebugInfo) -> DebugInfo {
    use super::{CubeClRuntimeConfig, RuntimeConfig};
    use core::sync::atomic::{AtomicU8, Ordering};

    const UNREAD: u8 = u8::MAX;
    const NO_LIMIT: u8 = u8::MAX - 1;
    static LIMIT: AtomicU8 = AtomicU8::new(UNREAD);

    let mut limit = LIMIT.load(Ordering::Relaxed);
    if limit == UNREAD {
        limit = CubeClRuntimeConfig::get()
            .compilation
            .debug_info
            .map_or(NO_LIMIT, |level| level as u8);
        LIMIT.store(limit, Ordering::Relaxed);
    }
    // Decode with the casts that encoded the level, so the order of the variants does not matter.
    let limit = [DebugInfo::None, DebugInfo::LineTables, DebugInfo::Full]
        .into_iter()
        .find(|level| *level as u8 == limit);
    resolve_debug_info(requested, limit)
}

/// At least [`PROFILE_DEBUG_INFO`], at most `limit`.
pub fn resolve_debug_info(requested: DebugInfo, limit: Option<DebugInfo>) -> DebugInfo {
    let level = requested.max(PROFILE_DEBUG_INFO);
    limit.map_or(level, |limit| level.min(limit))
}
