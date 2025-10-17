use crate::{
    memory_management::{BytesFormat, MemoryUsage},
    server::IoError,
    storage::{ComputeStorage, StorageHandle, StorageUtilization},
};

use alloc::vec::Vec;

use super::{MemoryPool, Slice, SliceBinding, SliceHandle, calculate_padding};

/// A memory pool that allocates buffers in a range of sizes and reuses them to minimize allocations.
///
/// - Only one slice is supported per page, due to the limitations in WGPU where each buffer should only bound with
///   either read only or read_write slices but not a mix of both.
/// - The pool uses a ring buffer to efficiently manage and reuse pages.
pub struct ExclusiveMemoryPool {
    pages: Vec<MemoryPage>,
    alignment: u64,
    dealloc_period: u64,
    last_dealloc_check: u64,
    max_alloc_size: u64,
    cur_avg_size: f64,
}

impl core::fmt::Display for ExclusiveMemoryPool {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!(
            " - Exclusive Pool max_alloc_size={}\n",
            BytesFormat::new(self.max_alloc_size)
        ))?;

        for page in self.pages.iter() {
            let is_free = page.slice.is_free();
            let size = BytesFormat::new(page.slice.effective_size());

            f.write_fmt(format_args!("   - Page {size} is_free={is_free}\n"))?;
        }

        if !self.pages.is_empty() {
            f.write_fmt(format_args!("\n{}\n", self.get_memory_usage()))?;
        }

        Ok(())
    }
}

const SIZE_AVG_DECAY: f64 = 0.01;

// How many times to find the allocation 'free' before deallocating it.
const ALLOC_AFTER_FREE: u32 = 5;

struct MemoryPage {
    slice: Slice,
    alloc_size: u64,
    free_count: u32,
}

impl ExclusiveMemoryPool {
    pub(crate) fn new(max_alloc_size: u64, alignment: u64, dealloc_period: u64) -> Self {
        // Pages should be allocated to be aligned.
        assert_eq!(max_alloc_size % alignment, 0);

        Self {
            pages: Vec::new(),
            alignment,
            dealloc_period,
            last_dealloc_check: 0,
            max_alloc_size,
            cur_avg_size: max_alloc_size as f64 / 2.0,
        }
    }

    /// Finds a free page that can contain the given size
    /// Returns a slice on that page if successful.
    fn get_free_page(&mut self, size: u64) -> Option<&mut MemoryPage> {
        // Return the smallest free page that fits.
        self.pages
            .iter_mut()
            .filter(|page| page.alloc_size >= size && page.slice.is_free())
            .min_by_key(|page| page.free_count)
    }

    fn alloc_page<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
    ) -> Result<&mut MemoryPage, IoError> {
        let alloc_size = (self.cur_avg_size as u64)
            .max(size)
            .next_multiple_of(self.alignment);

        let storage = storage.alloc(alloc_size)?;

        let handle = SliceHandle::new();
        let padding = calculate_padding(size, self.alignment);
        let mut slice = Slice::new(storage, handle, padding);

        // Return a smaller part of the slice. By construction, we only ever
        // get a page with a big enough size, so this is ok to do.
        slice.storage.utilization = StorageUtilization { offset: 0, size };
        slice.padding = padding;

        self.pages.push(MemoryPage {
            slice,
            alloc_size,
            // Start the allocation at 'almost ready to free'. Every use will decrement this.
            // This means allocations start as "suspected as unused" and over time will be kept for longer.
            free_count: ALLOC_AFTER_FREE - 1,
        });

        let idx = self.pages.len() - 1;
        Ok(&mut self.pages[idx])
    }
}

impl MemoryPool for ExclusiveMemoryPool {
    /// Returns the resource from the storage, for the specified handle.
    fn get(&self, binding: &SliceBinding) -> Option<&StorageHandle> {
        let binding_id = *binding.id();
        self.pages
            .iter()
            .find(|page| page.slice.id() == binding_id)
            .map(|page| &page.slice.storage)
    }

    /// Reserves memory of specified size using the reserve algorithm, and return
    /// a handle to the reserved memory.
    ///
    /// Also clean ups, merging free slices together if permitted by the merging strategy
    fn try_reserve(&mut self, size: u64) -> Option<SliceHandle> {
        self.cur_avg_size =
            self.cur_avg_size * (1.0 - SIZE_AVG_DECAY) + size as f64 * SIZE_AVG_DECAY;

        let padding = calculate_padding(size, self.alignment);

        self.get_free_page(size).map(|page| {
            // Return a smaller part of the slice. By construction, we only ever
            // get a page with a big enough size, so this is ok to do.
            page.slice.storage.utilization = StorageUtilization { offset: 0, size };
            page.slice.padding = padding;
            page.free_count = page.free_count.saturating_sub(1);
            page.slice.handle.clone()
        })
    }

    fn alloc<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
    ) -> Result<SliceHandle, IoError> {
        if size > self.max_alloc_size {
            return Err(IoError::BufferTooBig(size as usize));
        }

        let page = self.alloc_page(storage, size)?;
        Ok(page.slice.handle.clone())
    }

    fn get_memory_usage(&self) -> MemoryUsage {
        let used_slices: Vec<_> = self
            .pages
            .iter()
            .filter(|page| !page.slice.is_free())
            .collect();

        MemoryUsage {
            number_allocs: used_slices.len() as u64,
            bytes_in_use: used_slices
                .iter()
                .map(|page| page.slice.storage.size())
                .sum(),
            bytes_padding: used_slices.iter().map(|page| page.slice.padding).sum(),
            bytes_reserved: self.pages.iter().map(|page| page.alloc_size).sum(),
        }
    }

    fn max_alloc_size(&self) -> u64 {
        self.max_alloc_size
    }

    fn cleanup<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        alloc_nr: u64,
        explicit: bool,
    ) {
        // Check such that an alloc is free after at most dealloc_period.
        let check_period = self.dealloc_period / (ALLOC_AFTER_FREE as u64);

        if explicit || alloc_nr - self.last_dealloc_check >= check_period {
            self.last_dealloc_check = alloc_nr;

            self.pages.retain_mut(|page| {
                if page.slice.is_free() {
                    page.free_count += 1;

                    // If free found is sufficiently high (ie. we've seen this alloc as free multiple times,
                    // without it being used in the meantime), deallocate it.
                    if page.free_count >= ALLOC_AFTER_FREE || explicit {
                        storage.dealloc(page.slice.storage.id);
                        return false;
                    }
                }
                true
            });
        }
    }
}
