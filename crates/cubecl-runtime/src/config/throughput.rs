/// Whether throughput measurements are cached.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ThroughputConfig {
    /// Whether every measurement probes again, neither reading nor storing a cached one, in
    /// memory or on disk.
    #[serde(default)]
    pub disable_cache: bool,
}
