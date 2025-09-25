/// TestGen macro to test virtual memory pool functionality.
#[macro_export]
macro_rules! testgen_virtual_memory_pool {
    // Generate test modules for different configurations
    ([$($config_name:ident: {
        min_alloc: $min_alloc:expr,
        max_alloc: $max_alloc:expr,
        alignment: $alignment:expr
    }),*]) => {
        #[allow(non_snake_case)]
        mod virtual_memory_pool_tests {
            use super::*;

            // Ensure TestStorage is defined in scope
            fn _check_test_storage() {
                let _: TestStorage = panic!("This function is never called, just for type checking");
            }

            ::paste::paste! {
                $(mod [<$config_name _tests>] {
                    use super::*;
                    $crate::testgen_virtual_memory_pool!(
                        $config_name,
                        $min_alloc,
                        $max_alloc,
                        $alignment
                    );
                })*
            }
        }
    };

    // Single configuration without array
    ($min_alloc:expr, $max_alloc:expr, $alignment:expr) => {
        $crate::testgen_virtual_memory_pool!(default: {
            min_alloc: $min_alloc,
            max_alloc: $max_alloc,
            alignment: $alignment
        });
    };

    // Generate individual test suite for a specific configuration
    ($config_name:ident, $min_alloc:expr, $max_alloc:expr, $alignment:expr) => {
        const MIN_ALLOC: u64 = $min_alloc;
        const MAX_ALLOC: u64 = $max_alloc;
        const ALIGNMENT: u64 = $alignment;

        // Helper function to create storage instance - uses the TestStorage type defined in scope
        fn create_test_storage() -> TestStorage {
            TestStorage::default()
        }

        #[cfg(test)]
        mod basic_operations {
            use super::*;

            #[test]
            fn test_pool_creation() {
                let pool = VirtualMemoryPool::new(MIN_ALLOC, MAX_ALLOC, ALIGNMENT);
                assert_eq!(pool.max_alloc_size(), MAX_ALLOC);

                let usage = pool.get_memory_usage();
                assert_eq!(usage.number_allocs, 0);
                assert_eq!(usage.bytes_in_use, 0);
                assert_eq!(usage.bytes_reserved, 0);
            }

            #[test]
            fn test_single_allocation() -> Result<(), IoError> {
                let mut pool = VirtualMemoryPool::new(MIN_ALLOC, MAX_ALLOC, ALIGNMENT);
                let mut storage = create_test_storage();

                let handle = pool.alloc(1024, 0, &mut storage)?;
                assert!(handle.id().0 > 0);

                let usage = pool.get_memory_usage();
                assert_eq!(usage.number_allocs, 1);
                assert!(usage.bytes_in_use >= 1024);

                Ok(())
            }

            #[test]
            fn test_multiple_allocations() -> Result<(), IoError> {
                let mut pool = VirtualMemoryPool::new(MIN_ALLOC, MAX_ALLOC, ALIGNMENT);
                let mut storage = create_test_storage();

                let sizes = [512, 1024, 2048, 4096];
                let mut handles = Vec::new();

                for (i, &size) in sizes.iter().enumerate() {
                    let handle = pool.alloc(size, i, &mut storage)?;
                    handles.push(handle);
                }

                let usage = pool.get_memory_usage();
                assert_eq!(usage.number_allocs, sizes.len() as u64);
                assert!(usage.bytes_in_use >= sizes.iter().sum::<u64>());

                Ok(())
            }

            #[test]
            fn test_allocation_with_binding() -> Result<(), IoError> {
                let mut pool = VirtualMemoryPool::new(MIN_ALLOC, MAX_ALLOC, ALIGNMENT);
                let mut storage = create_test_storage();

                let handle = pool.alloc(2048, 0, &mut storage)?;
                let binding = SliceBinding::new(handle.id());

                let storage_handle = pool.get(&binding);
                assert!(storage_handle.is_some());
                assert!(storage_handle.unwrap().size() >= 2048);

                Ok(())
            }
        }

        #[cfg(test)]
        mod reservation_tests {
            use super::*;

            #[test]
            fn test_try_reserve_after_alloc() -> Result<(), IoError> {
                let mut pool = VirtualMemoryPool::new(MIN_ALLOC, MAX_ALLOC, ALIGNMENT);
                let mut storage = create_test_storage();

                // First allocate something large to create a page with potential free space
                let _handle1 = pool.alloc(MAX_ALLOC / 4, 0, &mut storage)?;

                // Try to reserve from existing space
                let handle2 = pool.try_reserve(1024, &mut storage);

                // This may or may not succeed depending on the storage implementation
                // The important thing is that it doesn't panic
                if handle2.is_some() {
                    let usage = pool.get_memory_usage();
                    assert_eq!(usage.number_allocs, 2);
                }

                Ok(())
            }

            #[test]
            fn test_try_reserve_no_space() {
                let mut pool = VirtualMemoryPool::new(MIN_ALLOC, MAX_ALLOC, ALIGNMENT);
                let mut storage = create_test_storage();

                // Try to reserve without any allocated pages
                let handle = pool.try_reserve(1024, &mut storage);
                assert!(handle.is_none(), "Should not be able to reserve from empty pool");
            }

            #[test]
            fn test_reservation_after_defrag() -> Result<(), IoError> {
                let mut pool = VirtualMemoryPool::new(MIN_ALLOC, MAX_ALLOC, ALIGNMENT);
                let mut storage = create_test_storage();

                // Allocate enough to trigger defragmentation
                for i in 0..25 {
                    let _handle = pool.alloc(ALIGNMENT, i, &mut storage)?;
                }

                // Try reservation after defrag
                let handle = pool.try_reserve(ALIGNMENT, &mut storage);
                // Result depends on storage implementation

                Ok(())
            }
        }

        #[cfg(test)]
        mod storage_specific_tests {
            use super::*;

            #[test]
            fn test_storage_granularity_alignment() -> Result<(), IoError> {
                let mut pool = VirtualMemoryPool::new(MIN_ALLOC, MAX_ALLOC, ALIGNMENT);
                let mut storage = create_test_storage();

                let storage_alignment = storage.granularity() as u64;

                // Test allocation with storage's natural granularity
                let handle = pool.alloc(storage_alignment * 2, 0, &mut storage)?;
                assert!(handle.id().0 > 0);

                let usage = pool.get_memory_usage();
                assert_eq!(usage.bytes_in_use, storage_alignment * 2);

                Ok(())
            }

            #[test]
            fn test_unaligned_size_handling() -> Result<(), IoError> {
                let mut pool = VirtualMemoryPool::new(MIN_ALLOC, MAX_ALLOC, ALIGNMENT);
                let mut storage = create_test_storage();

                // Test odd size that needs padding
                let odd_size = ALIGNMENT + 13;
                let handle = pool.alloc(odd_size, 0, &mut storage)?;
                assert!(handle.id().0 > 0);

                let usage = pool.get_memory_usage();
                assert!(usage.bytes_padding > 0);
                assert_eq!(usage.bytes_in_use, odd_size);

                Ok(())
            }

            #[test]
            fn test_storage_are_aligned_method() -> Result<(), IoError> {
                let mut pool = VirtualMemoryPool::new(MIN_ALLOC, MAX_ALLOC, ALIGNMENT);
                let mut storage = create_test_storage();

                let storage_granularity = storage.granularity() as u64;

                // Allocate two blocks
                let handle1 = pool.alloc(storage_granularity, 0, &mut storage)?;
                let handle2 = pool.alloc(storage_granularity, 1, &mut storage)?;

                let binding1 = SliceBinding::new(handle1.id());
                let binding2 = SliceBinding::new(handle2.id());

                if let (Some(storage1), Some(storage2)) = (pool.get(&binding1), pool.get(&binding2)) {
                    // Test storage's are_aligned method
                    let aligned = storage.are_aligned(&storage1.id, &storage2.id);
                    // Result depends on storage implementation
                    println!("Storage alignment check result: {}", aligned);
                }

                Ok(())
            }
        }

        #[cfg(test)]
        mod defragmentation_tests {
            use super::*;

            #[test]
            fn test_defragmentation_trigger() -> Result<(), IoError> {
                let mut pool = VirtualMemoryPool::new(MIN_ALLOC, MAX_ALLOC, ALIGNMENT);
                let mut storage = create_test_storage();

                // Allocate enough times to trigger defragmentation (DEFRAG_THRESHOLD = 20)
                for i in 0..25 {
                    let _handle = pool.alloc(ALIGNMENT * 2, i, &mut storage)?;
                }

                let usage = pool.get_memory_usage();
                assert_eq!(usage.number_allocs, 25);

                Ok(())
            }

            #[test]
            fn test_explicit_cleanup() -> Result<(), IoError> {
                let mut pool = VirtualMemoryPool::new(MIN_ALLOC, MAX_ALLOC, ALIGNMENT);
                let mut storage = create_test_storage();

                let _handle = pool.alloc(ALIGNMENT * 4, 0, &mut storage)?;
                let usage_before = pool.get_memory_usage();

                // Test explicit cleanup
                pool.cleanup(&mut storage, 1, true);

                // Pool should still be functional after cleanup
                let _handle2 = pool.alloc(ALIGNMENT * 2, 1, &mut storage)?;
                let usage_after = pool.get_memory_usage();

                assert!(usage_after.number_allocs >= 1);

                Ok(())
            }

            #[test]
            fn test_memory_compaction() -> Result<(), IoError> {
                let mut pool = VirtualMemoryPool::new(MIN_ALLOC, MAX_ALLOC, ALIGNMENT);
                let mut storage = create_test_storage();

                // Create fragmented memory pattern
                let mut handles = Vec::new();
                for i in 0..10 {
                    let handle = pool.alloc(ALIGNMENT, i, &mut storage)?;
                    handles.push(handle);
                }

                let fragmented_usage = pool.get_memory_usage();

                // Trigger defragmentation by allocating more
                for i in 10..25 {
                    let _handle = pool.alloc(ALIGNMENT, i, &mut storage)?;
                }

                let compacted_usage = pool.get_memory_usage();
                assert_eq!(compacted_usage.number_allocs, 25);

                Ok(())
            }
        }

        #[cfg(test)]
        mod memory_usage_tests {
            use super::*;

            #[test]
            fn test_memory_usage_accuracy() -> Result<(), IoError> {
                let mut pool = VirtualMemoryPool::new(MIN_ALLOC, MAX_ALLOC, ALIGNMENT);
                let mut storage = create_test_storage();

                let initial_usage = pool.get_memory_usage();
                assert_eq!(initial_usage.number_allocs, 0);

                let alloc_size = 1337; // Unaligned size to test padding
                let _handle = pool.alloc(alloc_size, 0, &mut storage)?;

                let after_alloc_usage = pool.get_memory_usage();
                assert_eq!(after_alloc_usage.number_allocs, 1);
                assert_eq!(after_alloc_usage.bytes_in_use, alloc_size);
                assert!(after_alloc_usage.bytes_padding > 0); // Should have padding
                assert!(after_alloc_usage.bytes_reserved >= after_alloc_usage.bytes_in_use + after_alloc_usage.bytes_padding);

                Ok(())
            }

            #[test]
            fn test_usage_tracking_multiple_allocs() -> Result<(), IoError> {
                let mut pool = VirtualMemoryPool::new(MIN_ALLOC, MAX_ALLOC, ALIGNMENT);
                let mut storage = create_test_storage();

                let sizes = [100, 200, 300, 400, 500];
                let total_requested: u64 = sizes.iter().sum();

                for (i, &size) in sizes.iter().enumerate() {
                    let _handle = pool.alloc(size, i, &mut storage)?;
                }

                let usage = pool.get_memory_usage();
                assert_eq!(usage.number_allocs, sizes.len() as u64);
                assert_eq!(usage.bytes_in_use, total_requested);

                Ok(())
            }

            #[test]
            fn test_memory_usage_with_reservations() -> Result<(), IoError> {
                let mut pool = VirtualMemoryPool::new(MIN_ALLOC, MAX_ALLOC, ALIGNMENT);
                let mut storage = create_test_storage();

                // Make an allocation first
                let _handle1 = pool.alloc(2048, 0, &mut storage)?;

                let usage_after_alloc = pool.get_memory_usage();

                // Try a reservation
                if let Some(_handle2) = pool.try_reserve(1024, &mut storage) {
                    let usage_after_reserve = pool.get_memory_usage();
                    assert!(usage_after_reserve.number_allocs > usage_after_alloc.number_allocs);
                }

                Ok(())
            }
        }

        #[cfg(test)]
        mod stress_tests {
            use super::*;

            #[test]
            fn test_many_small_allocations() -> Result<(), IoError> {
                let mut pool = VirtualMemoryPool::new(MIN_ALLOC, MAX_ALLOC, ALIGNMENT);
                let mut storage = create_test_storage();

                let num_allocs = 50;
                let small_size = ALIGNMENT;

                for i in 0..num_allocs {
                    let _handle = pool.alloc(small_size, i, &mut storage)?;
                }

                let usage = pool.get_memory_usage();
                assert_eq!(usage.number_allocs, num_allocs);

                Ok(())
            }

            #[test]
            fn test_mixed_workload() -> Result<(), IoError> {
                let mut pool = VirtualMemoryPool::new(MIN_ALLOC, MAX_ALLOC, ALIGNMENT);
                let mut storage = create_test_storage();

                // Mix of small, medium, and large allocations
                let small_size = ALIGNMENT;
                let medium_size = ALIGNMENT * 16;
                let large_size = ALIGNMENT * 64;

                let mut total_allocs = 0;

                // Small allocations
                for i in 0..20 {
                    let _handle = pool.alloc(small_size, total_allocs + i, &mut storage)?;
                }
                total_allocs += 20;

                // Medium allocations
                for i in 0..10 {
                    let _handle = pool.alloc(medium_size, total_allocs + i, &mut storage)?;
                }
                total_allocs += 10;

                // Large allocations
                for i in 0..5 {
                    let _handle = pool.alloc(large_size, total_allocs + i, &mut storage)?;
                }
                total_allocs += 5;

                let usage = pool.get_memory_usage();
                assert_eq!(usage.number_allocs, total_allocs);

                Ok(())
            }

            #[test]
            fn test_allocation_patterns() -> Result<(), IoError> {
                let mut pool = VirtualMemoryPool::new(MIN_ALLOC, MAX_ALLOC, ALIGNMENT);
                let mut storage = create_test_storage();

                // Test different allocation patterns
                let patterns = [
                    ALIGNMENT,
                    ALIGNMENT * 2,
                    ALIGNMENT * 3,
                    ALIGNMENT + 1, // Unaligned
                    ALIGNMENT * 10,
                ];

                for (i, &size) in patterns.iter().enumerate() {
                    let handle = pool.alloc(size, i, &mut storage)?;
                    assert!(handle.id().0 > 0);
                }

                let usage = pool.get_memory_usage();
                assert_eq!(usage.number_allocs, patterns.len() as u64);

                Ok(())
            }
        }

        #[cfg(test)]
        mod edge_cases {
            use super::*;

            #[test]
            fn test_minimum_allocation() -> Result<(), IoError> {
                let mut pool = VirtualMemoryPool::new(MIN_ALLOC, MAX_ALLOC, ALIGNMENT);
                let mut storage = create_test_storage();

                let handle = pool.alloc(1, 0, &mut storage)?;
                assert!(handle.id().0 > 0);

                let usage = pool.get_memory_usage();
                assert!(usage.bytes_padding > 0); // Should be padded to alignment

                Ok(())
            }

            #[test]
            fn test_exact_alignment_size() -> Result<(), IoError> {
                let mut pool = VirtualMemoryPool::new(MIN_ALLOC, MAX_ALLOC, ALIGNMENT);
                let mut storage = create_test_storage();

                let handle = pool.alloc(ALIGNMENT, 0, &mut storage)?;
                assert!(handle.id().0 > 0);

                let usage = pool.get_memory_usage();
                assert_eq!(usage.bytes_in_use, ALIGNMENT);

                Ok(())
            }

            #[test]
            fn test_zero_size_allocation() -> Result<(), IoError> {
                let mut pool = VirtualMemoryPool::new(MIN_ALLOC, MAX_ALLOC, ALIGNMENT);
                let mut storage = create_test_storage();

                let handle = pool.alloc(0, 0, &mut storage)?;
                assert!(handle.id().0 > 0);

                let usage = pool.get_memory_usage();
                assert_eq!(usage.bytes_in_use, 0);
                assert!(usage.bytes_padding >= ALIGNMENT); // Should be padded

                Ok(())
            }

            #[test]
            #[should_panic(expected = "Invalid allocation size")]
            fn test_oversized_physical_allocation() {
                let mut pool = VirtualMemoryPool::new(MIN_ALLOC, MAX_ALLOC, ALIGNMENT);
                let mut storage = create_test_storage();

                // This should panic in get_or_alloc_physical due to size validation
                let _ = pool.alloc(MAX_ALLOC + ALIGNMENT, 0, &mut storage);
            }

            #[test]
            fn test_boundary_size_allocation() -> Result<(), IoError> {
                let mut pool = VirtualMemoryPool::new(MIN_ALLOC, MAX_ALLOC, ALIGNMENT);
                let mut storage = create_test_storage();

                // Test allocation at boundaries
                if MIN_ALLOC > 0 {
                    let handle = pool.alloc(MIN_ALLOC, 0, &mut storage)?;
                    assert!(handle.id().0 > 0);
                }

                Ok(())
            }
        }

        #[cfg(test)]
        mod integration_tests {
            use super::*;

            #[test]
            fn test_full_workflow() -> Result<(), IoError> {
                let mut pool = VirtualMemoryPool::new(MIN_ALLOC, MAX_ALLOC, ALIGNMENT);
                let mut storage = create_test_storage();

                // 1. Initial allocation
                let handle1 = pool.alloc(ALIGNMENT * 2, 0, &mut storage)?;

                // 2. Verify we can get binding
                let binding1 = SliceBinding::new(handle1.id());
                let storage_handle1 = pool.get(&binding1);
                assert!(storage_handle1.is_some());

                // 3. More allocations to test scaling
                let mut handles = vec![handle1];
                for i in 1..15 {
                    let handle = pool.alloc(ALIGNMENT, i, &mut storage)?;
                    handles.push(handle);
                }

                // 4. Verify memory usage
                let usage = pool.get_memory_usage();
                assert_eq!(usage.number_allocs, 15);

                // 5. Test reservation
                let _reserved_handle = pool.try_reserve(ALIGNMENT, &mut storage);

                // 6. Trigger cleanup
                pool.cleanup(&mut storage, 20, true);

                // 7. Verify pool still works
                let _final_handle = pool.alloc(ALIGNMENT, 100, &mut storage)?;

                Ok(())
            }

            #[test]
            fn test_interleaved_operations() -> Result<(), IoError> {
                let mut pool = VirtualMemoryPool::new(MIN_ALLOC, MAX_ALLOC, ALIGNMENT);
                let mut storage = create_test_storage();

                let mut handles = Vec::new();

                // Interleave allocations and reservations
                for i in 0..10 {
                    // Allocate
                    let handle = pool.alloc(ALIGNMENT * (i + 1), i * 2, &mut storage)?;
                    handles.push(handle);

                    // Try to reserve
                    if let Some(reserved) = pool.try_reserve(ALIGNMENT, &mut storage) {
                        handles.push(reserved);
                    }
                }

                let usage = pool.get_memory_usage();
                assert!(usage.number_allocs >= 10);

                Ok(())
            }
        }
    };
}
