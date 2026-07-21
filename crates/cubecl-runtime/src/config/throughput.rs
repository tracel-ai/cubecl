#[cfg(std_io)]
use super::cache::CacheConfig;

/// Configuration for throughput in `CubeCL`.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ThroughputConfig {
    /// Cache location for storing throughput results.
    #[serde(default)]
    #[cfg(std_io)]
    pub cache: CacheConfig,

    /// Whether to enable caching of throughput results.
    #[serde(default)]
    pub disable_cache: bool,
}
