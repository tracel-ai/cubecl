use super::compilation::PROFILE_DEBUG_INFO;
use super::{CubeClRuntimeConfig, RuntimeConfig};
use core::sync::atomic::{AtomicU8, Ordering};
use cubecl_ir::settings::DebugInfo;

/// [`LIMIT`] before the configuration was read.
const UNREAD: u8 = u8::MAX;
/// [`LIMIT`] when the configuration sets no limit.
const NO_LIMIT: u8 = u8::MAX - 1;
/// The configured limit on debug data, encoded as its `u8`, read once.
static LIMIT: AtomicU8 = AtomicU8::new(UNREAD);

/// The debug data of a kernel that asks for `requested`, with the global configuration: the same
/// level as [`CompilationConfig::resolve_debug_info`](super::compilation::CompilationConfig::resolve_debug_info).
/// It reads the configuration once, so a kernel id can call it at each launch.
pub fn effective_debug_info(requested: DebugInfo) -> DebugInfo {
    let limit = match LIMIT.load(Ordering::Relaxed) {
        UNREAD => read_limit(),
        limit => limit,
    };
    resolve_debug_info(requested, decode_limit(limit))
}

/// At least [`PROFILE_DEBUG_INFO`], at most `limit`.
pub fn resolve_debug_info(requested: DebugInfo, limit: Option<DebugInfo>) -> DebugInfo {
    let level = requested.max(PROFILE_DEBUG_INFO);
    limit.map_or(level, |limit| level.min(limit))
}

/// The configured limit, encoded, kept in [`LIMIT`] for the next call.
fn read_limit() -> u8 {
    let limit = CubeClRuntimeConfig::get()
        .compilation
        .debug_info
        .map_or(NO_LIMIT, |level| level as u8);
    LIMIT.store(limit, Ordering::Relaxed);
    limit
}

/// The level `limit` encodes, decoded with the casts that encoded it, so the order of the
/// variants does not matter. `None` for [`NO_LIMIT`].
fn decode_limit(limit: u8) -> Option<DebugInfo> {
    [DebugInfo::None, DebugInfo::LineTables, DebugInfo::Full]
        .into_iter()
        .find(|level| *level as u8 == limit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_level_round_trips_its_encoding() {
        assert_eq!(
            decode_limit(DebugInfo::LineTables as u8),
            Some(DebugInfo::LineTables)
        );
    }

    #[test]
    fn no_limit_decodes_to_none() {
        assert_eq!(decode_limit(NO_LIMIT), None);
    }

    #[test]
    fn resolving_keeps_at_least_the_profile_level() {
        assert_eq!(
            resolve_debug_info(DebugInfo::None, None),
            PROFILE_DEBUG_INFO
        );
    }

    #[test]
    fn resolving_keeps_what_the_kernel_asks_for() {
        assert_eq!(resolve_debug_info(DebugInfo::Full, None), DebugInfo::Full);
    }

    #[test]
    fn a_limit_lowers_the_level() {
        assert_eq!(
            resolve_debug_info(DebugInfo::Full, Some(DebugInfo::LineTables)),
            DebugInfo::LineTables
        );
    }

    #[test]
    fn a_limit_never_raises_the_level() {
        assert_eq!(
            resolve_debug_info(DebugInfo::None, Some(DebugInfo::Full)),
            PROFILE_DEBUG_INFO
        );
    }
}
