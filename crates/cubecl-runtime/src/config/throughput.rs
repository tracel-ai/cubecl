/// Whether throughput measurements are cached.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ThroughputConfig {
    /// Whether measured values are kept off disk, neither read from nor stored to it. How long a
    /// value is kept in memory is [`ThroughputCache`](crate::throughput::ThroughputCache)'s to say.
    #[serde(default)]
    pub disable_cache: bool,
}
