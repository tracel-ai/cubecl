//! End-to-end test of the programmatic pool layout: the whole path from a
//! [`MemoryPoolsConfig`] payload through [`MemoryConfiguration::resolve`] to
//! the actual pool behavior, plus the in-place rebuild
//! ([`MemoryManagement::configure`]) that re-sizes the pools between
//! workloads. There is deliberately no config-file pathway for pool layouts —
//! they are dynamic, set at runtime per workload.

use std::sync::Arc;

use cubecl_ir::MemoryDeviceProperties;
use cubecl_runtime::config::memory::{MemoryPoolConfig, MemoryPoolsConfig};
use cubecl_runtime::config::size::MemorySize;
use cubecl_runtime::logging::ServerLogger;
use cubecl_runtime::memory_management::{
    MemoryConfiguration, MemoryManagement, MemoryManagementOptions,
};
use cubecl_runtime::storage::BytesStorage;

const MIB: u64 = 1024 * 1024;

fn sliced(page_size: u64, pages: u64) -> MemoryPoolsConfig {
    MemoryPoolsConfig::Explicit(vec![MemoryPoolConfig::Sliced {
        page_size: MemorySize(page_size),
        max_slice_size: None,
        max_pool_size: Some(MemorySize(page_size * pages)),
        dealloc_period: None,
    }])
}

fn props() -> MemoryDeviceProperties {
    MemoryDeviceProperties {
        max_page_size: 128 * MIB,
        alignment: 32,
    }
}

#[test]
fn programmatic_pools_override_runtime_default() {
    // The path a downstream user takes when the budget is computed at runtime
    // (e.g. an LLM activation working set), and what every runtime does for
    // its main GPU pool at stream creation.
    let pools = sliced(MIB, 2);
    let resolved = MemoryConfiguration::default()
        .resolve(Some(&pools), &props())
        .unwrap();
    let mut memory_management = MemoryManagement::from_configuration(
        BytesStorage::default(),
        &props(),
        resolved,
        Arc::new(ServerLogger::default()),
        MemoryManagementOptions::new("Main GPU Memory"),
    );

    // Vastly different sizes share the same arena instead of landing in
    // size-bucketed pools with separate reservations.
    let small = memory_management.reserve(4096).unwrap();
    drop(small);
    let _large = memory_management.reserve(512 * 1024).unwrap();
    assert_eq!(memory_management.memory_usage().bytes_reserved, MIB);

    // The budget is a hard cap.
    let _fill_1 = memory_management.reserve(MIB).unwrap();
    let _fill_2 = memory_management.reserve(500 * 1024).unwrap();
    assert!(memory_management.reserve(MIB).is_err());
    assert_eq!(memory_management.memory_usage().bytes_reserved, 2 * MIB);
}

#[test]
fn configure_rebuilds_pools_in_place() {
    let resolved = MemoryConfiguration::default()
        .resolve(Some(&sliced(MIB, 2)), &props())
        .unwrap();
    let mut memory_management = MemoryManagement::from_configuration(
        BytesStorage::default(),
        &props(),
        resolved,
        Arc::new(ServerLogger::default()),
        MemoryManagementOptions::new("Main GPU Memory"),
    );

    // While an allocation is live, the rebuild is refused and the old layout
    // (2 × 1 MiB cap) stays in force.
    let live = memory_management.reserve(MIB).unwrap();
    let bigger = MemoryConfiguration::default()
        .resolve(Some(&sliced(4 * MIB, 2)), &props())
        .unwrap();
    assert!(!memory_management.configure(bigger.clone(), &props()));
    assert!(memory_management.reserve(2 * MIB).is_err());

    // At a quiescent point the rebuild goes through, and the new layout
    // serves what the old cap refused.
    drop(live);
    assert!(memory_management.configure(bigger, &props()));
    let _large = memory_management.reserve(2 * MIB).unwrap();
}
