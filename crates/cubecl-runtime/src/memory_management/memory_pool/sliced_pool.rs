use crate::{
    memory_management::{
        BytesFormat, MemoryUsage,
        memory_pool::{MemoryPage, MemoryPool},
    },
    storage::StorageId,
};
use alloc::vec::Vec;
use core::fmt::Display;
use hashbrown::HashMap;

pub struct SlicedPool {
    pages: HashMap<StorageId, MemoryPage>,
    page_size: u64,
    alignment: u64,
    max_alloc_size: u64,
}

impl SlicedPool {
    pub fn new(page_size: u64, max_slice_size: u64, alignment: u64) -> Self {
        Self {
            pages: HashMap::new(),
            page_size,
            alignment,
            max_alloc_size: max_slice_size,
        }
    }
}

impl MemoryPool for SlicedPool {
    fn accept(&self, size: u64) -> bool {
        self.max_alloc_size >= size
            ||
            // If the size is close to the page size so it doesn't create much fragmentation with
            // unused space.
            match self.page_size.checked_sub(size) {
                Some(diff) => diff * 5 < self.page_size, // 20 % unused space is the max allowed.
                None => false,
            }
    }

    fn get(&self, binding: &super::SliceBinding) -> Option<&crate::storage::StorageHandle> {
        for (_, page) in self.pages.iter() {
            if let Some(handle) = page.get(binding) {
                return Some(handle);
            }
        }

        None
    }

    fn try_reserve(&mut self, size: u64) -> Option<super::SliceHandle> {
        for (_, page) in self.pages.iter_mut() {
            page.coalesce();
            if let Some(handle) = page.try_reserve(size) {
                return Some(handle);
            }
        }

        None
    }

    fn alloc<Storage: crate::storage::ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
    ) -> Result<super::SliceHandle, crate::server::IoError> {
        let storage = storage.alloc(self.page_size)?;
        let storage_id = storage.id;
        let mut page = MemoryPage::new(storage, self.alignment);
        let returned = page.try_reserve(size);
        self.pages.insert(storage_id, page);

        Ok(returned.expect("effective_size to be smaller than page_size"))
    }

    fn get_memory_usage(&self) -> MemoryUsage {
        let mut usage = MemoryUsage {
            number_allocs: 0,
            bytes_in_use: 0,
            bytes_padding: 0,
            bytes_reserved: 0,
        };

        for (_, page) in self.pages.iter() {
            let current = page.memory_usage();
            usage = usage.combine(current);
        }

        usage
    }

    fn cleanup<Storage: crate::storage::ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        _alloc_nr: u64,
        explicit: bool,
    ) {
        if !explicit {
            return;
        }
        let mut to_clean = Vec::new();

        for (id, page) in self.pages.iter_mut() {
            page.coalesce();
            let summary = page.summary(false);
            if summary.amount_free == summary.amount_total {
                to_clean.push(*id);
            }
        }

        for id in to_clean {
            self.pages.remove(&id);
            storage.dealloc(id);
        }
    }
}

impl Display for SlicedPool {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.pages.is_empty() {
            return Ok(());
        }

        f.write_fmt(format_args!(
            " - Sliced Pool page_size={} max_alloc_size={}\n",
            BytesFormat::new(self.page_size),
            BytesFormat::new(self.max_alloc_size)
        ))?;

        for (id, page) in self.pages.iter() {
            let summary = page.summary(false);
            f.write_fmt(format_args!(
                "   - Page {id} num_slices={} =>",
                summary.num_total
            ))?;

            let size_free = BytesFormat::new(summary.amount_free);
            let size_full = BytesFormat::new(summary.amount_full);
            let size_total = BytesFormat::new(summary.amount_total);

            f.write_fmt(format_args!(
                " {size_free} free - {size_full} full - {size_total} total\n"
            ))?;
        }

        f.write_fmt(format_args!("\n{}\n", self.get_memory_usage()))?;

        Ok(())
    }
}
