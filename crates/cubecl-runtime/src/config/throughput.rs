/// Whether throughput measurements are cached.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ThroughputConfig {
    /// Whether measurements are kept off disk: each key is still probed once a process, and its
    /// answer kept in memory for the rest of it, but none is read from or stored to disk.
    #[serde(default)]
    pub disable_cache: bool,
}
