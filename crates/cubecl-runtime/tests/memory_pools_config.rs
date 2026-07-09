//! End-to-end test of the `memory.pools` global config override: the whole
//! path from `CubeClRuntimeConfig::set()` through `MemoryConfiguration::resolve`
//! to the actual pool behavior.
//!
//! This lives in its own integration-test binary because the global config is
//! a per-process singleton: `set()` must run before any `get()` and only once.

use std::sync::Arc;

use cubecl_ir::MemoryDeviceProperties;
use cubecl_runtime::config::memory::{MemoryPoolConfig, MemoryPoolsConfig};
use cubecl_runtime::config::size::MemorySize;
use cubecl_runtime::config::{CubeClRuntimeConfig, RuntimeConfig};
use cubecl_runtime::logging::ServerLogger;
use cubecl_runtime::memory_management::{
    MemoryConfiguration, MemoryManagement, MemoryManagementOptions,
};
use cubecl_runtime::storage::BytesStorage;

const MIB: u64 = 1024 * 1024;

#[test]
fn global_pools_config_overrides_runtime_default() {
    // The programmatic path a downstream user takes when the budget is
    // computed at runtime (e.g. an LLM KV cache).
    let mut config = CubeClRuntimeConfig::default();
    config.memory.pools = Some(MemoryPoolsConfig::Explicit(vec![MemoryPoolConfig::Sliced {
        page_size: MemorySize(MIB),
        max_slice_size: None,
        max_pool_size: Some(MemorySize(2 * MIB)),
        preallocate: true,
        dealloc_period: None,
    }]));
    CubeClRuntimeConfig::set(config);

    let props = MemoryDeviceProperties {
        max_page_size: 128 * MIB,
        alignment: 32,
    };

    // What every runtime does for its main GPU pool at server creation.
    let resolved = MemoryConfiguration::default().resolve(&props);
    let mut memory_management = MemoryManagement::from_configuration(
        BytesStorage::default(),
        &props,
        resolved,
        Arc::new(ServerLogger::default()),
        MemoryManagementOptions::new("Main GPU Memory"),
    );

    // Preallocated: the footprint is fixed before any reservation.
    assert_eq!(memory_management.memory_usage().bytes_reserved, 2 * MIB);

    // Vastly different sizes share the same arena instead of landing in
    // size-bucketed pools with separate reservations.
    let small = memory_management.reserve(4096).unwrap();
    drop(small);
    let _large = memory_management.reserve(512 * 1024).unwrap();
    assert_eq!(memory_management.memory_usage().bytes_reserved, 2 * MIB);

    // The budget is a hard cap.
    let _fill_1 = memory_management.reserve(MIB).unwrap();
    let _fill_2 = memory_management.reserve(500 * 1024).unwrap();
    assert!(memory_management.reserve(MIB).is_err());
    assert_eq!(memory_management.memory_usage().bytes_reserved, 2 * MIB);
}
