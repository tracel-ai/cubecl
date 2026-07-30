/// Configuration for throughput in `CubeCL`.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ThroughputConfig {
    /// Whether to enable caching of throughput results.
    #[serde(default)]
    pub disable_cache: bool,
}
