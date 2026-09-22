//! The adaptive pool's statistic across environments: recorded in the one
//! active while the pool grew, and adopted — or dropped — when another is
//! activated. Its own binary, since switching the environment is global.

use cubecl_environment::environment;
use cubecl_environment::sync::Arc;
use cubecl_ir::MemoryDeviceProperties;
use cubecl_server::logging::ServerLogger;
use cubecl_server::memory_management::{
    ErrorGraph, MemoryConfiguration, MemoryManagement, MemoryManagementOptions, MemoryPoolKind,
    MemoryPoolOptions, PoolType,
};
use cubecl_server::storage::BytesStorage;

const MIB: u64 = 1024 * 1024;

fn adaptive() -> MemoryManagement<BytesStorage> {
    MemoryManagement::from_configuration(
        BytesStorage::default(),
        &MemoryDeviceProperties::new(1024 * MIB, 32),
        MemoryConfiguration::Custom {
            pool_options: vec![MemoryPoolOptions {
                pool_type: PoolType::AdaptivePages {
                    min_page_size: 4 * MIB,
                },
                dealloc_period: None,
            }],
        },
        Arc::new(ServerLogger::default()),
        MemoryManagementOptions::new("adaptive"),
    )
}

/// The adaptive pool's page size and its outdated pages.
fn pool(memory: &MemoryManagement<BytesStorage>) -> (u64, u64) {
    match memory.memory_report().dynamic[0].kind {
        MemoryPoolKind::Adaptive {
            page_size,
            outdated_pages,
        } => (page_size, outdated_pages),
        _ => unreachable!("the only pool is adaptive"),
    }
}

#[test]
fn a_switch_adopts_the_new_environments_page_size() {
    let root = tempfile::tempdir().unwrap();
    // Loading the runtime config activates the configured environment, so it
    // is loaded first — or it would undo the redirect on first use.
    <cubecl_server::config::CubeClRuntimeConfig as cubecl_server::config::RuntimeConfig>::get();
    environment::set_root(root.path());
    let failures = &mut ErrorGraph::default();

    environment::activate("large");
    let mut memory = adaptive();
    let _large = memory.reserve(10 * MIB, failures).unwrap();
    assert_eq!(pool(&memory), (11 * MIB, 0));

    // An environment that never ran the workload: back to the floor, and the
    // page held is outdated from the next reservation on.
    environment::activate("fresh");
    let _small = memory.reserve(MIB, failures).unwrap();
    assert_eq!(pool(&memory), (4 * MIB, 1));

    // Back to the one that recorded it: a new pool starts there, and the live
    // pool adopts it on its next reservation — without the small allocation
    // it just served overwriting the record.
    environment::activate("large");
    assert_eq!(pool(&adaptive()), (11 * MIB, 0));
    let _tick = memory.reserve(MIB, failures).unwrap();
    assert_eq!(pool(&memory).0, 11 * MIB);
    assert_eq!(pool(&adaptive()), (11 * MIB, 0), "the record survived");
}
