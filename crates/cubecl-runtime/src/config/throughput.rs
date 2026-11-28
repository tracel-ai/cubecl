/// Whether throughput measurements are cached.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ThroughputConfig {
    /// Whether measurements are kept off disk: each key is still probed once a process, and its
    /// answer kept in memory until its cache is cleared, but none is read from or stored to disk.
    #[serde(default)]
    pub disable_cache: bool,
}
