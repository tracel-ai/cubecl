use super::logger::{LogLevel, LoggerConfig};

/// Configuration for memory settings in CubeCL.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, Default)]
pub struct MemoryConfig {
    /// Logger configuration for memory-related logs, using specific log levels.
    #[serde(default)]
    pub logger: LoggerConfig<MemoryLogLevel>,
    /// Configuration for persistent memory pools.
    #[serde(default)]
    pub persistent_memory: PersistentMemory,
}

/// Configuration options for persistent memory pools in CubeCL runtimes.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, Default)]
pub enum PersistentMemory {
    /// Persistent memory is enabled but used only when explicitly specified.
    #[default]
    Enabled,
    /// Persistent memory is disabled, allowing dynamic allocations.
    Disabled,
    /// Persistent memory is enforced, preventing dynamic allocations.
    ///
    /// # Warning
    ///
    /// Enforcing persistent memory may cause out-of-memory errors if tensors of varying sizes are used.
    Enforced,
}

/// Log levels for memory-related events in CubeCL.
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
