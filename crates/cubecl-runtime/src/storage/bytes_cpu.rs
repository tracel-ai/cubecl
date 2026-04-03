use crate::server::IoError;

use super::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use alloc::alloc::{Layout, alloc_zeroed, dealloc};
use cubecl_common::backtrace::BackTrace;
use hashbrown::HashMap;

/// The bytes storage maps ids to pointers of bytes in a contiguous layout.
#[derive(Default)]
pub struct BytesStorage {
    memory: HashMap<StorageId, AllocatedBytes>,
}

impl core::fmt::Debug for BytesStorage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("BytesStorage")
    }
}

/// Can send to other threads.
unsafe impl Send for BytesStorage {}
unsafe impl Send for BytesResource {}

/// This struct is a pointer to a memory chunk or slice.
#[derive(Debug)]
pub struct BytesResource {
    ptr: *mut u8,
    utilization: StorageUtilization,
}

/// This struct refers to a specific (contiguous) layout of bytes.
struct AllocatedBytes {
    ptr: *mut u8,
    layout: Layout,
}

impl BytesResource {
    /// Returns a mutable pointer to the start of the resource and its length.
    pub fn get_write_ptr_and_length(&self) -> (*mut u8, usize) {
        (
            // SAFETY:
            // - The offset is created to be within the bounds of the allocation.
            unsafe { self.ptr.add(self.utilization.offset as usize) },
            self.utilization.size as usize,
        )
    }

    /// Returns the resource as a mutable slice of bytes.
    ///
    /// The lifetime `'a` is the lifetime of the underlying `BytesStorage` allocation,
    /// not of `self`. The `&mut self` ensures only one mutable slice is created per
    /// resource. Multiple resources may point to non-overlapping regions of the same
    /// allocation (like `split_at_mut`); each resource owns its region exclusively.
    pub fn write<'a>(&mut self) -> &'a mut [u8] {
        let (ptr, len) = self.get_write_ptr_and_length();

        // SAFETY:
        // - ptr is non-null and aligned (from BytesStorage::alloc).
        // - The region [ptr..ptr+len) is within a single allocation.
        // - Memory is initialized (BytesStorage uses alloc_zeroed).
        // - `&mut self` ensures exclusive access to this resource's region.
        // - `StorageHandle` assigns non-overlapping regions per resource.
        // - Systems must make sure this is the only `BytesResource` with an outstanding mutable borrow.
        unsafe { core::slice::from_raw_parts_mut(ptr, len) }
    }

    /// Returns the resource as an immutable slice of bytes.
    ///
    /// See [`write`](Self::write) for lifetime and safety notes.
    pub fn read<'a>(&self) -> &'a [u8] {
        let (ptr, len) = self.get_write_ptr_and_length();

        // SAFETY:
        // - ptr is non-null and aligned (from BytesStorage::alloc).
        // - The region [ptr..ptr+len) is within a single allocation.
        // - Memory is initialized (BytesStorage uses alloc_zeroed).
        unsafe { core::slice::from_raw_parts(ptr, len) }
    }
}

impl ComputeStorage for BytesStorage {
    type Resource = BytesResource;

    fn alignment(&self) -> usize {
        4
    }

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let allocated_bytes = self.memory.get(&handle.id).unwrap();

        BytesResource {
            ptr: allocated_bytes.ptr,
            utilization: handle.utilization.clone(),
        }
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, size))
    )]
    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        let id = StorageId::new();
        let handle = StorageHandle {
            id,
            utilization: StorageUtilization { offset: 0, size },
        };

        if size == 0 {
            // Zero-size allocations are valid handles but don't need real memory.
            let memory = AllocatedBytes {
                ptr: core::ptr::NonNull::dangling().as_ptr(),
                layout: Layout::new::<()>(),
            };
            self.memory.insert(id, memory);
        } else {
            unsafe {
                let layout = Layout::array::<u8>(size as usize).unwrap();

                // We allocate zeroed memory since we expose it as &[u8] / &mut [u8]
                // which requires initialization.
                let ptr = alloc_zeroed(layout);
                if ptr.is_null() {
                    return Err(IoError::BufferTooBig {
                        size,
                        backtrace: BackTrace::capture(),
                    });
                }
                let memory = AllocatedBytes { ptr, layout };
                self.memory.insert(id, memory);
            }
        }

        Ok(handle)
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    fn dealloc(&mut self, id: StorageId) {
        if let Some(memory) = self.memory.remove(&id)
            && memory.layout.size() > 0
        {
            unsafe {
                dealloc(memory.ptr, memory.layout);
            }
        }
    }

    fn flush(&mut self) {
        // We don't wait for dealloc.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_log::test]
    fn test_can_alloc_and_dealloc() {
        let mut storage = BytesStorage::default();
        let handle_1 = storage.alloc(64).unwrap();

        assert_eq!(handle_1.size(), 64);
        storage.dealloc(handle_1.id);
    }

    #[test_log::test]
    fn test_slices() {
        let mut storage = BytesStorage::default();
        let handle_1 = storage.alloc(64).unwrap();
        let handle_2 = StorageHandle::new(
            handle_1.id,
            StorageUtilization {
                offset: 24,
                size: 8,
            },
        );

        storage
            .get(&handle_1)
            .write()
            .iter_mut()
            .enumerate()
            .for_each(|(i, b)| {
                *b = i as u8;
            });

        let bytes = storage.get(&handle_2).read().to_vec();

        storage.dealloc(handle_1.id);
        assert_eq!(bytes, &[24, 25, 26, 27, 28, 29, 30, 31]);
    }

    /// Miri catches: "reading memory, but memory is uninitialized"
    #[test_log::test]
    fn test_read_after_alloc_without_write() {
        let mut storage = BytesStorage::default();
        let handle = storage.alloc(16).unwrap();
        let resource = storage.get(&handle);
        assert!(resource.read().iter().all(|&b| b == 0));
        storage.dealloc(handle.id);
    }

    /// Miri catches: "creating allocation with size 0"
    #[test_log::test]
    fn test_zero_size_alloc_and_dealloc() {
        let mut storage = BytesStorage::default();
        let handle = storage.alloc(0).unwrap();
        assert_eq!(handle.size(), 0);
        storage.dealloc(handle.id);
    }

    #[test_log::test]
    fn test_alloc_dealloc_realloc() {
        let mut storage = BytesStorage::default();
        let h1 = storage.alloc(32).unwrap();
        storage.get(&h1).write()[0] = 0xAA;
        storage.dealloc(h1.id);
        let h2 = storage.alloc(32).unwrap();
        storage.dealloc(h2.id);
    }

    #[test_log::test]
    fn test_multiple_non_overlapping_regions() {
        let mut storage = BytesStorage::default();
        let base = storage.alloc(64).unwrap();

        let regions: alloc::vec::Vec<_> = (0..4)
            .map(|i| {
                StorageHandle::new(
                    base.id,
                    StorageUtilization {
                        offset: i * 16,
                        size: 16,
                    },
                )
            })
            .collect();

        for (i, region) in regions.iter().enumerate() {
            storage.get(region).write().fill(i as u8);
        }
        for (i, region) in regions.iter().enumerate() {
            assert!(storage.get(region).read().iter().all(|&b| b == i as u8));
        }
        storage.dealloc(base.id);
    }
}
