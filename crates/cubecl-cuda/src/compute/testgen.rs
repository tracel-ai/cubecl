// One strange thing I have noticed is than when runnning this tests with 1 test thread some of them fail due to OOM.
// running them altogeteher solves the issue. Need to review the root cause of this.
#[macro_export]
macro_rules! testgen_storage {
   ($storage_type:ty, $create_fn:ident) => {
       paste::paste! {
           mod [<$storage_type:lower _tests>] {
               use super::*;
               use $crate::compute::{CudaStorage, ExpandableStorage, CudaStorageType};
               use cubecl_runtime::storage::ComputeStorage;
               use cubecl_core::server::IoError;
               use cudarc::driver::sys::*;
               use cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS;
               use std::ptr;

               /// Sets up a CUDA context for testing
               /// Returns (context, device_id) tuple for test cleanup
               fn setup_cuda_context() -> (CUcontext, i32) {
                   unsafe {
                       assert_eq!(cuInit(0), CUDA_SUCCESS);
                       let mut device: CUdevice = 0;
                       assert_eq!(cuDeviceGet(&mut device, 0), CUDA_SUCCESS);
                       let mut context: CUcontext = ptr::null_mut();
                       assert_eq!(cuCtxCreate_v2(&mut context, 0, device), CUDA_SUCCESS);
                       (context, 0)
                   }
               }

               /// Creates a storage instance using the specified creation function
               fn $create_fn(device_id: i32) -> $storage_type {
                   testgen_storage!(@create_storage $storage_type, device_id)
               }

               /// Tests basic allocation functionality
               /// Verifies that allocations return correct sizes, offsets, and unique IDs
               #[test]
               fn [<test_ $storage_type:lower _alloc>]() {
                   let (context, device_id) = setup_cuda_context();
                   unsafe { cuCtxSetCurrent(context); }

                   let mut storage = $create_fn(device_id);
                   let size = 512 * 1024; // 512KB test allocation
                   let handle = storage.alloc(size).expect("Failed to allocate memory");

                   // Verify allocation properties
                   assert_eq!(handle.size(), size);
                   assert_eq!(handle.offset(), 0);

                   // Test second allocation to ensure unique IDs
                   let handle2 = storage.alloc(size).expect("Failed to allocate second block");
                   assert_eq!(handle2.size(), size);
                   assert_ne!(handle.id, handle2.id);

                   unsafe { cuCtxDestroy_v2(context); }
               }

               /// Tests deallocation queuing behavior (lazy deallocation pattern)
               /// Verifies that memory remains accessible until explicit flush
               #[test]
               fn [<test_ $storage_type:lower _dealloc>]() {
                   let (context, device_id) = setup_cuda_context();
                   unsafe { cuCtxSetCurrent(context); }

                   let mut storage = $create_fn(device_id);
                   let size = 512 * 1024; // 512KB test allocation
                   let handle1 = storage.alloc(size).expect("Failed to allocate");
                   let handle2 = storage.alloc(size).expect("Failed to allocate");

                   // Verify resources are accessible before deallocation
                   let _resource1 = storage.get(&handle1);
                   let _resource2 = storage.get(&handle2);

                   // Mark for deallocation (lazy pattern)
                   storage.dealloc(handle1.id);
                   storage.dealloc(handle2.id);

                   // Verify deallocation is queued but not executed
                   testgen_storage!(@verify_dealloc $storage_type, storage, 2);

                   // Memory should still be accessible until flush
                   let _resource1 = storage.get(&handle1);

                   unsafe { cuCtxDestroy_v2(context); }
               }

               /// Tests flush functionality and memory cleanup
               /// Verifies that flush actually releases queued deallocations
               #[test]
               fn [<test_ $storage_type:lower _flush>]() {
                   let (context, device_id) = setup_cuda_context();
                   unsafe { cuCtxSetCurrent(context); }

                   let mut storage = $create_fn(device_id);
                   let size = 512 * 1024; // 512KB test allocation
                   let handle1 = storage.alloc(size).expect("Failed to allocate");
                   let handle2 = storage.alloc(size).expect("Failed to allocate");

                   // Queue deallocations
                   storage.dealloc(handle1.id);
                   storage.dealloc(handle2.id);
                   testgen_storage!(@verify_dealloc $storage_type, storage, 2);

                   // Execute flush and verify cleanup
                   storage.flush();
                   testgen_storage!(@verify_flush $storage_type, storage, handle1, handle2);

                   unsafe { cuCtxDestroy_v2(context); }
               }

               /// Tests resource retrieval functionality
               /// Verifies that resources have correct properties and valid pointers
               #[test]
               fn [<test_ $storage_type:lower _get_resource>]() {
                   let (context, device_id) = setup_cuda_context();
                   unsafe { cuCtxSetCurrent(context); }

                   let mut storage = $create_fn(device_id);
                   let size = 512 * 1024; // 512KB test allocation
                   let handle = storage.alloc(size).expect("Failed to allocate");

                   // Retrieve and verify resource properties
                   let resource = storage.get(&handle);
                   assert_eq!(resource.offset(), 0);
                   assert_eq!(resource.size(), size);
                   assert!(resource.ptr != 0); // Valid GPU pointer

                   unsafe { cuCtxDestroy_v2(context); }
               }
           }
       }
   };

   // Internal helper: Create storage instances based on type
   (@create_storage CudaStorage, $device_id:expr) => {
       {
           let stream = ptr::null_mut();
           let mem_alignment = 32;
           CudaStorage::new(mem_alignment, stream)
       }
   };

   (@create_storage ExpandableStorage, $device_id:expr) => {
       {
           let stream = ptr::null_mut();
           let virtual_size = 1024 * 1024 * 1024; // 1GB virtual space
           let handle_size = 2 * 1024 * 1024; // 2MB per handle
           ExpandableStorage::new($device_id, stream, virtual_size, handle_size, handle_size)
       }
   };

   // Internal helper: Verify deallocation queuing based on storage type
   (@verify_dealloc CudaStorage, $storage:expr, $expected:expr) => {
       // CudaStorage uses deallocations Vec for lazy deallocation
       assert_eq!($storage.deallocations_len(), $expected);
   };

   (@verify_dealloc ExpandableStorage, $storage:expr, $expected:expr) => {
       // ExpandableStorage also uses deallocations Vec for lazy deallocation
       assert_eq!($storage.deallocations_len(), $expected);
   };

   // Internal helper: Verify flush behavior based on storage type
   (@verify_flush CudaStorage, $storage:expr, $handle1:expr, $handle2:expr) => {
       assert_eq!($storage.deallocations_len(), 0);
       assert!(!$storage.memory_contains_key(&$handle1.id));
       assert!(!$storage.memory_contains_key(&$handle2.id));
   };

   (@verify_flush ExpandableStorage, $storage:expr, $handle1:expr, $handle2:expr) => {
       assert_eq!($storage.deallocations_len(), 0);
       assert!(!$storage.memory_contains_key(&$handle1.id));
       assert!(!$storage.memory_contains_key(&$handle2.id));
   };
}



#[macro_export]
macro_rules! testgen_storage_advanced {
   ($storage_type:ty, $create_fn:ident) => {
       paste::paste! {
           mod [<$storage_type:lower _advanced_tests>] {
               use super::*;
               use $crate::compute::{CudaStorage, ExpandableStorage};
               use cubecl_runtime::storage::ComputeStorage;
               use cubecl_core::server::IoError;
               use cudarc::driver::sys::*;
               use cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS;
               use std::ptr;

               /// Sets up a CUDA context for advanced testing
               fn setup_cuda_context() -> (CUcontext, i32) {
                   unsafe {
                       assert_eq!(cuInit(0), CUDA_SUCCESS);
                       let mut device: CUdevice = 0;
                       assert_eq!(cuDeviceGet(&mut device, 0), CUDA_SUCCESS);
                       let mut context: CUcontext = ptr::null_mut();
                       assert_eq!(cuCtxCreate_v2(&mut context, 0, device), CUDA_SUCCESS);
                       (context, 0)
                   }
               }

               /// Creates a storage instance for advanced testing
               fn $create_fn(device_id: i32) -> $storage_type {
                   testgen_storage!(@create_storage $storage_type, device_id)
               }

               /// Tests complex allocation/deallocation lifecycle scenarios
               /// Verifies partial deallocation, memory reuse, and state consistency
               #[test]
               fn [<test_ $storage_type:lower _lifecycle>]() {
                   let (context, device_id) = setup_cuda_context();
                   unsafe { cuCtxSetCurrent(context); }

                   let mut storage = $create_fn(device_id);
                   let base_size = 256 * 1024; // 256KB base allocation

                   // Allocate multiple blocks with varying sizes
                   let handles: Vec<_> = (0..5)
                       .map(|i| {
                           let size = (i + 1) * base_size;
                           storage.alloc(size as u64).expect("Failed to allocate")
                       })
                       .collect();

                   // Verify all allocations succeeded
                   assert_eq!(handles.len(), 5);
                   testgen_storage_advanced!(@verify_memory_count $storage_type, storage, 5);

                   // Deallocate selective blocks (partial deallocation)
                   storage.dealloc(handles[1].id);
                   storage.dealloc(handles[3].id);
                   testgen_storage!(@verify_dealloc $storage_type, storage, 2);

                   // Execute flush and verify selective cleanup
                   storage.flush();

                   // Verify only deallocated blocks are removed
                   testgen_storage_advanced!(@verify_memory_count $storage_type, storage, 3);
                   testgen_storage_advanced!(@verify_handle_exists $storage_type, storage, handles[0].id, true);
                   testgen_storage_advanced!(@verify_handle_exists $storage_type, storage, handles[1].id, false);
                   testgen_storage_advanced!(@verify_handle_exists $storage_type, storage, handles[2].id, true);
                   testgen_storage_advanced!(@verify_handle_exists $storage_type, storage, handles[3].id, false);
                   testgen_storage_advanced!(@verify_handle_exists $storage_type, storage, handles[4].id, true);

                   unsafe { cuCtxDestroy_v2(context); }
               }

               // Generate storage-specific large allocation and error handling tests
               testgen_storage_advanced!(@large_alloc_test $storage_type, $create_fn);
               testgen_storage_advanced!(@error_handling_test $storage_type, $create_fn);
           }
       }
   };

   // Storage-specific large allocation test for ExpandableStorage
   (@large_alloc_test ExpandableStorage, $create_fn:ident) => {
       /// Tests large allocations that span multiple handles in ExpandableStorage
       /// Verifies correct handle calculation and memory mapping
       #[test]
       fn test_expandable_large_allocation() {
           let (context, device_id) = setup_cuda_context();
           unsafe { cuCtxSetCurrent(context); }

           let mut storage = $create_fn(device_id);
           let large_size = 6 * 1024 * 1024; // 6MB (should span 3 handles of 2MB each)
           let handle = storage.alloc(large_size).expect("Failed to allocate large block");

           assert_eq!(handle.size(), large_size);

           // Verify the block spans the expected number of handles
           let block = storage.get_block(&handle.id).unwrap();
           assert_eq!(block.len(), 3); // 6MB / 2MB = 3 handles

           unsafe { cuCtxDestroy_v2(context); }
       }
   };

   // Storage-specific large allocation test for CudaStorage
   (@large_alloc_test CudaStorage, $create_fn:ident) => {
       /// Tests large single allocations in CudaStorage
       /// Verifies that large contiguous allocations work correctly
       #[test]
       fn test_cuda_large_allocation() {
           let (context, device_id) = setup_cuda_context();
           unsafe { cuCtxSetCurrent(context); }

           let mut storage = $create_fn(device_id);
           let large_size = 64 * 1024 * 1024; // 64MB contiguous allocation
           let handle = storage.alloc(large_size).expect("Failed to allocate large block");

           assert_eq!(handle.size(), large_size);

           unsafe { cuCtxDestroy_v2(context); }
       }
   };

   // Error handling tests specific to ExpandableStorage
   (@error_handling_test ExpandableStorage, $create_fn:ident) => {
       /// Tests ExpandableStorage-specific error conditions
       /// Verifies virtual memory limits and handle pool constraints
       #[test]
       fn test_expandable_allocation_too_large() {
           let (context, device_id) = setup_cuda_context();
           unsafe { cuCtxSetCurrent(context); }

           let mut storage = $create_fn(device_id);
           let oversized = 2 * 1024 * 1024 * 1024; // 2GB > 1GB virtual_size limit
           let result = storage.alloc(oversized);

           // Should fail with BufferTooBig error due to virtual size limit
           assert!(result.is_err());
           if let Err(IoError::BufferTooBig(_)) = result {
               // Expected error type for exceeding virtual memory limit
           } else {
               panic!("Expected BufferTooBig error, got {:?}", result);
           }

           unsafe { cuCtxDestroy_v2(context); }
       }

       /// Tests handle pool capacity limits in ExpandableStorage
       #[test]
       fn test_expandable_handle_pool_limit() {
           let (context, device_id) = setup_cuda_context();
           unsafe { cuCtxSetCurrent(context); }

           let mut storage = $create_fn(device_id);
           let handle_size = 2 * 1024 * 1024; // 2MB per handle
           let virtual_size = 1024 * 1024 * 1024; // 1GB total virtual space
           let max_handles = virtual_size / handle_size; // 512 handles maximum

           // Try to allocate exactly at the theoretical limit
           let at_limit_size = max_handles * handle_size;
           let result = storage.alloc(at_limit_size);

           // Should work or fail gracefully depending on actual memory constraints
           match result {
               Ok(_) => {
                   // Successfully allocated at limit - this is valid behavior
               },
               Err(IoError::BufferTooBig(_)) => {
                   // Expected if we hit memory constraints before theoretical limit
               },
               Err(other) => {
                   panic!("Unexpected error: {:?}", other);
               }
           }

           unsafe { cuCtxDestroy_v2(context); }
       }
   };

   // Error handling tests specific to CudaStorage
   (@error_handling_test CudaStorage, $create_fn:ident) => {
       /// Tests CudaStorage-specific error conditions
       /// Verifies physical memory exhaustion and edge cases
       #[test]
       fn test_cuda_memory_exhaustion() {
           let (context, device_id) = setup_cuda_context();
           unsafe { cuCtxSetCurrent(context); }

           let mut storage = $create_fn(device_id);
           let mut successful_allocs = 0;
           let chunk_size = 100 * 1024 * 1024; // 100MB chunks
           let max_attempts = 50; // Reasonable limit to prevent infinite loops

           // Keep allocating until we hit physical memory limits
           for i in 0..max_attempts {
               match storage.alloc(chunk_size) {
                   Ok(_) => {
                       successful_allocs += 1;
                   },
                   Err(IoError::BufferTooBig(_)) => {
                       // Hit memory limit - this is expected behavior
                       break;
                   },
                   Err(other) => {
                       panic!("Unexpected error at allocation {}: {:?}", i, other);
                   }
               }
           }

           // Should have managed at least one allocation on any reasonable GPU
           assert!(successful_allocs > 0,
                  "Should have allocated at least some memory before hitting limits");

           unsafe { cuCtxDestroy_v2(context); }
       }

       /// Tests zero-size allocation behavior in CudaStorage
       #[test]
       fn test_cuda_zero_size_allocation() {
           let (context, device_id) = setup_cuda_context();
           unsafe { cuCtxSetCurrent(context); }

           let mut storage = $create_fn(device_id);

           // Test zero-size allocation edge case
           let result = storage.alloc(0);

           match result {
               Ok(handle) => {
                   assert_eq!(handle.size(), 0);
                   // Zero-size allocation should work in some implementations
               },
               Err(_) => {
                   // Some implementations might reject zero-size, which is also valid
               }
           }

           unsafe { cuCtxDestroy_v2(context); }
       }

       /// Tests alignment-related edge cases in CudaStorage
       #[test]
       fn test_cuda_alignment_edge_cases() {
           let (context, device_id) = setup_cuda_context();
           unsafe { cuCtxSetCurrent(context); }

           let mut storage = $create_fn(device_id);
           let alignment = storage.alignment();

           // Test allocation just below alignment boundary
           let size_below = alignment - 1;
           let handle1 = storage.alloc(size_below as u64).expect("Should handle size below alignment");
           assert_eq!(handle1.size(), size_below as u64);

           // Test allocation exactly at alignment boundary
           let size_at = alignment;
           let handle2 = storage.alloc(size_at as u64).expect("Should handle size at alignment");
           assert_eq!(handle2.size(), size_at as u64);

           // Test allocation just above alignment boundary
           let size_above = alignment + 1;
           let handle3 = storage.alloc(size_above as u64).expect("Should handle size above alignment");
           assert_eq!(handle3.size(), size_above as u64);

           unsafe { cuCtxDestroy_v2(context); }
       }
   };

   // Internal helpers for memory state verification
   (@verify_memory_count CudaStorage, $storage:expr, $expected:expr) => {
       assert_eq!($storage.memory_len(), $expected);
   };

   (@verify_memory_count ExpandableStorage, $storage:expr, $expected:expr) => {
       assert_eq!($storage.memory_len(), $expected);
   };

   (@verify_handle_exists CudaStorage, $storage:expr, $handle_id:expr, $should_exist:expr) => {
       assert_eq!($storage.memory_contains_key(&$handle_id), $should_exist);
   };

   (@verify_handle_exists ExpandableStorage, $storage:expr, $handle_id:expr, $should_exist:expr) => {
       assert_eq!($storage.memory_contains_key(&$handle_id), $should_exist);
   };
}
