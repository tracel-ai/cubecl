/// Whether throughput measurements are cached.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ThroughputConfig {
    /// Whether to measure throughput again in every process instead of reading the cached
    /// measurements.
    #[serde(default)]
    pub disable_cache: bool,
}
