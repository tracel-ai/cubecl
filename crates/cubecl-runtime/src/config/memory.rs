use super::logger::{LogLevel, LoggerConfig};

/// Configuration for memory settings in `CubeCL`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, Default)]
pub struct MemoryConfig {
    /// Logger configuration for memory-related logs, using specific log levels.
    #[serde(default)]
    pub logger: LoggerConfig<MemoryLogLevel>,
    /// Configuration for persistent memory pools.
    #[serde(default)]
    pub persistent_memory: PersistentMemory,
    /// Strategy used for the dynamic (activation) memory pools.
    #[serde(default)]
    pub dynamic_pool: DynamicPoolConfig,
}

/// Strategy used to build the dynamic memory pools that back activations.
///
/// The default ([`Auto`](DynamicPoolConfig::Auto)) keeps the
/// [`MemoryConfiguration`](crate::memory_management::MemoryConfiguration) chosen
/// by the runtime (e.g. `SubSlices`). [`SingleSliced`](DynamicPoolConfig::SingleSliced)
/// overrides it with a single coalescing arena so that allocations of every size
/// reuse the same chunks instead of each size-bucketed pool retaining its own
/// peak reservation.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, Default)]
#[serde(tag = "strategy", rename_all = "kebab-case")]
pub enum DynamicPoolConfig {
    /// Keep the [`MemoryConfiguration`](crate::memory_management::MemoryConfiguration)
    /// passed by the runtime (today's behavior).
    #[default]
    Auto,
    /// Route every dynamic allocation through a single sliced arena (plus the
    /// tiny sub-alignment exclusive pool). `page_size_bytes` is the per-chunk
    /// granularity; the pool allocates as many chunks as the live working set
    /// needs. It must be at least as large as the biggest single allocation, or
    /// that allocation is rejected. When omitted, a safe large default is used.
    SingleSliced {
        /// Per-chunk size in bytes. Aligned up to the device alignment.
        #[serde(default)]
        page_size_bytes: Option<u64>,
    },
}

/// Configuration options for persistent memory pools in `CubeCL` runtimes.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, Default)]
pub enum PersistentMemory {
    /// Persistent memory is enabled but used only when explicitly specified.
    #[default]
    #[serde(rename = "enabled")]
    Enabled,
    /// Persistent memory is disabled, allowing dynamic allocations.
    #[serde(rename = "disabled")]
    Disabled,
    /// Persistent memory is enforced, preventing dynamic allocations.
    ///
    /// # Warning
    ///
    /// Enforcing persistent memory may cause out-of-memory errors if tensors of varying sizes are used.
    #[serde(rename = "enforced")]
    Enforced,
}

/// Log levels for memory-related events in `CubeCL`.
#[derive(Default, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum MemoryLogLevel {
    /// No memory-related logging.
    #[default]
    #[serde(rename = "disabled")]
    Disabled,
    /// Logs basic memory events, such as creating memory pages and manually cleaning memory.
    #[serde(rename = "basic")]
    Basic,
    /// Logs detailed memory information.
    #[serde(rename = "full")]
    Full,
}

impl LogLevel for MemoryLogLevel {}
