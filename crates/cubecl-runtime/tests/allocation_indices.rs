//! Pool locations must survive allocation, binding and compaction past `u16::MAX`.
#![cfg(feature = "storage-bytes")]
#![forbid(unsafe_code)]

use cubecl_ir::MemoryDeviceProperties;
use cubecl_runtime::{
    logging::ServerLogger,
    memory_management::{
        ErrorGraph, ManagedMemoryHandle, MemoryConfiguration, MemoryManagement,
        MemoryManagementOptions, MemoryPoolOptions, PoolType,
    },
    storage::BytesStorage,
};
use std::{collections::HashSet, sync::Arc};

fn exercise(pool_type: PoolType) {
    let mut pool = MemoryManagement::from_configuration(
        BytesStorage::default(),
        &MemoryDeviceProperties {
            max_page_size: 32,
            alignment: 8,
        },
        MemoryConfiguration::Custom {
            pool_options: vec![MemoryPoolOptions {
                pool_type,
                dealloc_period: None,
            }],
        },
        Arc::new(ServerLogger::default()),
        MemoryManagementOptions::new("allocation index regression"),
    );
    let mut failures = ErrorGraph::default();
    let mut retained = Vec::new();
    let mut addresses = HashSet::new();
    for index in 0..65_538 {
        // A full page forces sliced pools across the boundary too. The first
        // allocation is smaller to catch both aliasing and an undersized result.
        let requested = if index == 0 { 8 } else { 32 };
        let reserved = pool.reserve(requested, &mut failures).unwrap();
        let assigned = ManagedMemoryHandle::new();
        pool.bind(reserved, assigned.clone(), 0, &mut failures)
            .unwrap();
        let resource = pool
            .get_resource(assigned.clone().binding(), None, None)
            .unwrap();
        let (pointer, length) = resource.get_write_ptr_and_length();
        assert_eq!(length, requested as usize, "allocation {index}");
        assert!(
            addresses.insert(pointer as usize),
            "allocation {index} aliases live storage"
        );
        retained.push((assigned, pointer as usize, length));
    }

    // Remove pages on both sides of the old boundary, compact, and verify the
    // surviving frontend handles still locate exactly their original storage.
    retained = retained
        .into_iter()
        .enumerate()
        .filter_map(|(i, value)| (i % 3 != 1).then_some(value))
        .collect();
    pool.cleanup(true, &mut failures);
    for (handle, address, length) in &retained {
        let resource = pool
            .get_resource(handle.clone().binding(), None, None)
            .unwrap();
        let (pointer, actual_length) = resource.get_write_ptr_and_length();
        assert_eq!(pointer as usize, *address);
        assert_eq!(actual_length, *length);
    }
    let extra = pool.reserve(32, &mut failures).unwrap();
    let resource = pool.get_resource(extra.binding(), None, None).unwrap();
    let (pointer, length) = resource.get_write_ptr_and_length();
    assert_eq!(length, 32);
    assert!(
        !retained
            .iter()
            .any(|(_, address, _)| *address == pointer as usize)
    );
}

#[test]
fn exclusive_pages_preserve_large_indices() {
    exercise(PoolType::ExclusivePages { max_alloc_size: 32 });
}

#[test]
#[cfg(not(exclusive_memory_only))]
fn sliced_pages_preserve_large_indices() {
    exercise(PoolType::SlicedPages {
        page_size: 32,
        max_slice_size: 32,
        max_pool_size: None,
    });
}

#[test]
#[cfg(not(exclusive_memory_only))]
fn configured_capacity_can_exceed_u16_pages() {
    exercise(PoolType::SlicedPages {
        page_size: 32,
        max_slice_size: 32,
        max_pool_size: Some(65_538 * 32),
    });
}
