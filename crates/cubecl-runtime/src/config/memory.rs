use super::logger::{LogLevel, LoggerConfig};

/// Configuration for memory settings in `CubeCL`.
///
/// Unknown fields are rejected so a leftover `pools` entry — the pools are the
/// memory management's to lay out, not a setting — or a misspelled option is a
/// load error rather than a silently dropped setting.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct MemoryConfig {
    /// Logger configuration for memory-related logs, using specific log levels.
    #[serde(default)]
    pub logger: LoggerConfig<MemoryLogLevel>,
    /// Configuration for persistent memory pools.
    #[serde(default)]
    pub persistent_memory: PersistentMemory,
}

/// Configuration options for persistent memory pools in `CubeCL` runtimes.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, Default)]
pub enum PersistentMemory {
    /// Persistent memory is enabled but used only when explicitly specified.
    #[default]
    #[serde(rename = "enabled")]
    Enabled,
    /// Like `enabled`, and automatic allocations whose size matches an
    /// existing persistent bucket are also served from the persistent pool.
    ///
    /// A good heuristic for training, where the recurring allocations are
    /// weight-shaped and updated within the same pool. Less suited to
    /// inference: activations that happen to match a weight size get pulled
    /// into exact-sized persistent slices.
    #[serde(rename = "size-match")]
    SizeMatch,
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

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn pools_rejected_in_config_files() {
        // Pool layouts are a programmatic setting; a leftover `pools` entry in
        // a config file must be a load error, not a silently ignored setting.
        assert!(toml::from_str::<MemoryConfig>("pools = \"sub-slices\"").is_err());
        assert!(
            toml::from_str::<MemoryConfig>("[[pools]]\ntype = \"sliced\"\npage_size = \"1MiB\"\n")
                .is_err()
        );
    }
}
