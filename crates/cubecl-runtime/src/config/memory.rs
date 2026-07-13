use super::logger::{LogLevel, LoggerConfig};
use super::size::MemorySize;
use alloc::vec::Vec;

/// Configuration for memory settings in `CubeCL`.
///
/// Unknown fields are rejected so a leftover `pools` entry (now a programmatic
/// setting, see [`MemoryPoolsConfig`]) or a misspelled option is a load error
/// rather than a silently dropped setting.
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

/// A pool layout override for a runtime's **main GPU** memory: a preset or an
/// explicit list of pool entries, tried in order at allocation time (the first
/// pool that accepts an allocation's size serves it).
///
/// This is a **programmatic** setting, deliberately not a config-file one —
/// pool layouts are dynamic (e.g. resized per model just before a load) and
/// must not freeze at startup. Apply it with
/// [`configure_memory_pools`](crate::client::ComputeClient::configure_memory_pools):
/// it rebuilds the calling stream's pools in place and becomes the layout for
/// streams created afterwards. Auxiliary pools (pinned CPU, staging, uniforms)
/// are never affected.
#[derive(Clone, Debug, PartialEq)]
pub enum MemoryPoolsConfig {
    /// A named preset matching the runtime-level presets.
    Preset(MemoryPoolsPreset),
    /// An explicit pool list, mirroring
    /// [`MemoryPoolOptions`](crate::memory_management::MemoryPoolOptions).
    Explicit(Vec<MemoryPoolConfig>),
}

/// The presets of [`MemoryConfiguration`](crate::memory_management::MemoryConfiguration).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryPoolsPreset {
    /// The runtime's `SubSlices` preset: a ladder of size-bucketed pools that
    /// sub-slice large pages.
    SubSlices,
    /// The runtime's `ExclusivePages` preset: one page per allocation, in
    /// exponentially spaced size buckets.
    ExclusivePages,
}

/// One pool entry; mirrors [`MemoryPoolOptions`](crate::memory_management::MemoryPoolOptions)
/// and [`PoolType`](crate::memory_management::PoolType).
///
/// Sizes are aligned up to the device alignment.
#[derive(Clone, Debug, PartialEq)]
pub enum MemoryPoolConfig {
    /// Every allocation gets its own page
    /// ([`PoolType::ExclusivePages`](crate::memory_management::PoolType::ExclusivePages)).
    Exclusive {
        /// Largest allocation this pool accepts. `0` is valid: it makes a pool
        /// dedicated to zero-sized (sub-alignment) allocations.
        max_alloc_size: MemorySize,
        /// Period (in parent allocation count) after which unused pages are
        /// deallocated. `None` never deallocates.
        dealloc_period: Option<u64>,
    },
    /// Allocations are slices of larger pages
    /// ([`PoolType::SlicedPages`](crate::memory_management::PoolType::SlicedPages)).
    Sliced {
        /// Size of each page.
        page_size: MemorySize,
        /// Largest slice this pool accepts. Defaults to `page_size`.
        max_slice_size: Option<MemorySize>,
        /// Hard cap on the pool's total reserved bytes: exceeding it is an
        /// error instead of silent growth. `None` grows unbounded. Note:
        /// runtimes that create one memory management per stream (CUDA, HIP)
        /// apply the cap per stream.
        max_pool_size: Option<MemorySize>,
        /// Period (in parent allocation count) after which unused pages are
        /// deallocated. `None` never deallocates.
        dealloc_period: Option<u64>,
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

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn pools_rejected_in_config_files() {
        // Pool layouts are a programmatic setting; a leftover `pools` entry in
        // a config file must be a load error, not a silently ignored setting.
        assert!(toml::from_str::<MemoryConfig>("pools = \"sub-slices\"").is_err());
        assert!(
            toml::from_str::<MemoryConfig>(
                "[[pools]]\ntype = \"sliced\"\npage_size = \"1MiB\"\n"
            )
            .is_err()
        );
    }
}
