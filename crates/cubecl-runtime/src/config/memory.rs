use super::logger::{LogLevel, LoggerConfig};
use super::size::MemorySize;
use alloc::vec::Vec;

/// Configuration for memory settings in `CubeCL`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, Default)]
pub struct MemoryConfig {
    /// Logger configuration for memory-related logs, using specific log levels.
    #[serde(default)]
    pub logger: LoggerConfig<MemoryLogLevel>,
    /// Configuration for persistent memory pools.
    #[serde(default)]
    pub persistent_memory: PersistentMemory,
    /// Overrides the pool layout of every runtime's **main GPU** memory.
    ///
    /// Omit to keep each runtime's own default. Either a preset name or an
    /// explicit `[[memory.pools]]` list. Auxiliary pools (pinned CPU, staging,
    /// uniforms) are never affected. An invalid layout (empty list, zero page
    /// size, slice larger than page) panics at server creation rather than
    /// being silently replaced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pools: Option<MemoryPoolsConfig>,
}

/// `memory.pools`: a preset name (`"sub-slices"` / `"exclusive-pages"`) or an
/// explicit list of pool entries, tried in order at allocation time (the first
/// pool that accepts an allocation's size serves it).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(
    untagged,
    expecting = "a preset string (\"sub-slices\" or \"exclusive-pages\") or a list of pool tables"
)]
pub enum MemoryPoolsConfig {
    /// A named preset matching the runtime-level presets.
    Preset(MemoryPoolsPreset),
    /// An explicit pool list, mirroring
    /// [`MemoryPoolOptions`](crate::memory_management::MemoryPoolOptions).
    Explicit(Vec<MemoryPoolConfig>),
}

/// The presets of [`MemoryConfiguration`](crate::memory_management::MemoryConfiguration).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
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
/// Sizes accept raw integers (bytes) or strings like `"8KiB"` / `"20GiB"`, and
/// are aligned up to the device alignment. Note that unknown fields are
/// ignored (a serde limitation of internally tagged enums), so double-check
/// field spellings.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum MemoryPoolConfig {
    /// Every allocation gets its own page
    /// ([`PoolType::ExclusivePages`](crate::memory_management::PoolType::ExclusivePages)).
    Exclusive {
        /// Largest allocation this pool accepts. `0` is valid: it makes a pool
        /// dedicated to zero-sized (sub-alignment) allocations.
        max_alloc_size: MemorySize,
        /// Period (in parent allocation count) after which unused pages are
        /// deallocated. Omit to never deallocate.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        dealloc_period: Option<u64>,
    },
    /// Allocations are slices of larger pages
    /// ([`PoolType::SlicedPages`](crate::memory_management::PoolType::SlicedPages)).
    Sliced {
        /// Size of each page.
        page_size: MemorySize,
        /// Largest slice this pool accepts. Defaults to `page_size`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_slice_size: Option<MemorySize>,
        /// Hard cap on the pool's total reserved bytes: exceeding it is an
        /// error instead of silent growth. Omit for unbounded growth.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_pool_size: Option<MemorySize>,
        /// Eagerly allocate all pages up to `max_pool_size` at server creation
        /// for a fixed footprint from startup. Requires `max_pool_size`. Note:
        /// runtimes that create one memory management per stream (CUDA, HIP)
        /// preallocate per stream.
        #[serde(default)]
        preallocate: bool,
        /// Period (in parent allocation count) after which unused pages are
        /// deallocated. Omit to never deallocate.
        #[serde(default, skip_serializing_if = "Option::is_none")]
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
    fn pools_omitted_keeps_none() {
        // Existing config files written before `pools` was added must continue
        // to deserialize unchanged.
        let config: MemoryConfig = toml::from_str("persistent_memory = \"enabled\"").unwrap();
        assert_eq!(config.pools, None);
    }

    #[test]
    fn pools_preset_string() {
        let config: MemoryConfig = toml::from_str("pools = \"sub-slices\"").unwrap();
        assert_eq!(
            config.pools,
            Some(MemoryPoolsConfig::Preset(MemoryPoolsPreset::SubSlices))
        );

        let config: MemoryConfig = toml::from_str("pools = \"exclusive-pages\"").unwrap();
        assert_eq!(
            config.pools,
            Some(MemoryPoolsConfig::Preset(MemoryPoolsPreset::ExclusivePages))
        );
    }

    #[test]
    fn pools_unknown_preset_fails() {
        assert!(toml::from_str::<MemoryConfig>("pools = \"bogus\"").is_err());
    }

    #[test]
    fn pools_explicit_list() {
        let config: MemoryConfig = toml::from_str(
            r#"
            [[pools]]
            type = "exclusive"
            max_alloc_size = "8KiB"
            dealloc_period = 10000

            [[pools]]
            type = "sliced"
            page_size = "20GiB"
            max_slice_size = "20GiB"
            max_pool_size = "20GiB"
            preallocate = true
            "#,
        )
        .unwrap();

        const GIB: u64 = 1024 * 1024 * 1024;
        let expected = MemoryPoolsConfig::Explicit(alloc::vec![
            MemoryPoolConfig::Exclusive {
                max_alloc_size: MemorySize(8 * 1024),
                dealloc_period: Some(10000),
            },
            MemoryPoolConfig::Sliced {
                page_size: MemorySize(20 * GIB),
                max_slice_size: Some(MemorySize(20 * GIB)),
                max_pool_size: Some(MemorySize(20 * GIB)),
                preallocate: true,
                dealloc_period: None,
            },
        ]);
        assert_eq!(config.pools, Some(expected));
    }

    #[test]
    fn pools_raw_integer_sizes_and_defaults() {
        let config: MemoryConfig = toml::from_str(
            r#"
            [[pools]]
            type = "sliced"
            page_size = 8192
            "#,
        )
        .unwrap();

        let expected = MemoryPoolsConfig::Explicit(alloc::vec![MemoryPoolConfig::Sliced {
            page_size: MemorySize(8192),
            max_slice_size: None,
            max_pool_size: None,
            preallocate: false,
            dealloc_period: None,
        }]);
        assert_eq!(config.pools, Some(expected));
    }

    #[test]
    fn pools_parse_from_nested_section() {
        // The untagged enum must also work one level down, where serde's
        // content buffering replays the values.
        let config: crate::config::CubeClRuntimeConfig = toml::from_str(
            r#"
            [memory]
            persistent_memory = "enabled"

            [[memory.pools]]
            type = "sliced"
            page_size = "1MiB"
            "#,
        )
        .unwrap();

        let expected = MemoryPoolsConfig::Explicit(alloc::vec![MemoryPoolConfig::Sliced {
            page_size: MemorySize(1024 * 1024),
            max_slice_size: None,
            max_pool_size: None,
            preallocate: false,
            dealloc_period: None,
        }]);
        assert_eq!(config.memory.pools, Some(expected));
    }

    #[test]
    fn pools_toml_roundtrip() {
        for pools in [
            MemoryPoolsConfig::Preset(MemoryPoolsPreset::SubSlices),
            MemoryPoolsConfig::Explicit(alloc::vec![
                MemoryPoolConfig::Exclusive {
                    max_alloc_size: MemorySize(0),
                    dealloc_period: None,
                },
                MemoryPoolConfig::Sliced {
                    page_size: MemorySize(4096),
                    max_slice_size: Some(MemorySize(2048)),
                    max_pool_size: Some(MemorySize(8192)),
                    preallocate: true,
                    dealloc_period: Some(500),
                },
            ]),
        ] {
            let config = MemoryConfig {
                pools: Some(pools),
                ..Default::default()
            };
            let serialized = toml::to_string(&config).unwrap();
            let parsed: MemoryConfig = toml::from_str(&serialized).unwrap();
            assert_eq!(parsed.pools, config.pools, "failed for: {serialized}");
        }
    }
}
