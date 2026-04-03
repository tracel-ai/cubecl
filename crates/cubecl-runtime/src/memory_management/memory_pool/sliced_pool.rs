use crate::{
    memory_management::{
        BytesFormat, ManagedMemoryHandle, MemoryLocation, MemoryUsage,
        memory_pool::{MemoryPage, MemoryPool, Slice},
    },
    server::IoError,
    storage::StorageId,
};
use alloc::vec::Vec;
use core::fmt::Display;

pub struct SlicedPool {
    pages: Vec<(MemoryPage, StorageId)>,
    pages_tmp: Vec<(MemoryPage, StorageId)>,
    page_size: u64,
    alignment: u64,
    max_alloc_size: u64,
    location_base: MemoryLocation,
}

impl SlicedPool {
    pub fn new(page_size: u64, max_slice_size: u64, alignment: u64, pool_pos: u8) -> Self {
        Self {
            pages: Vec::new(),
            pages_tmp: Vec::new(),
            page_size,
            alignment,
            max_alloc_size: max_slice_size,
            location_base: MemoryLocation::new(pool_pos, 0, 0),
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

    fn find(&self, binding: &super::ManagedMemoryBinding) -> Result<&Slice, IoError> {
        let (page, _) = &self.pages[binding.descriptor().page()];
        page.find(binding)
    }

    fn try_reserve(&mut self, size: u64) -> Option<super::ManagedMemoryHandle> {
        for (page, _) in self.pages.iter_mut() {
            page.coalesce();
            if let Some(handle) = page.try_reserve(size) {
                return Some(handle);
            }
        }

        None
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, storage))
    )]
    fn alloc<Storage: crate::storage::ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
    ) -> Result<super::ManagedMemoryHandle, crate::server::IoError> {
        let storage = storage.alloc(self.page_size)?;

        let storage_id = storage.id;
        let mut location_base = self.location_base;
        location_base.page = self.pages.len() as u16;

        let mut page = MemoryPage::new(storage, self.alignment, location_base);
        let returned = page.try_reserve(size);
        self.pages.push((page, storage_id));

        Ok(returned.expect("effective_size to be smaller than page_size"))
    }

    fn get_memory_usage(&self) -> MemoryUsage {
        let mut usage = MemoryUsage {
            number_allocs: 0,
            bytes_in_use: 0,
            bytes_padding: 0,
            bytes_reserved: 0,
        };

        for (page, _) in self.pages.iter() {
            let current = page.memory_usage();
            usage = usage.combine(current);
        }

        usage
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, storage))
    )]
    fn cleanup<Storage: crate::storage::ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        _alloc_nr: u64,
        explicit: bool,
    ) {
        if !explicit {
            return;
        }

        for (mut page, id) in self.pages.drain(..) {
            page.coalesce();
            let summary = page.summary(false);

            if summary.amount_free == summary.amount_total {
                storage.dealloc(id);
            } else {
                let page_pos = self.pages_tmp.len() as u16;
                page.update_page(page_pos);
                self.pages_tmp.push((page, id));
            }
        }

        core::mem::swap(&mut self.pages, &mut self.pages_tmp);
    }

    /// Binds a user defined [`ManagedMemoryHandle`] to a slice in this memory pool.
    fn bind(
        &mut self,
        reserved: ManagedMemoryHandle,
        assigned: ManagedMemoryHandle,
        cursor: u64,
    ) -> Result<(), IoError> {
        let (page, _) = &mut self.pages[reserved.descriptor().page()];

        page.bind(reserved, assigned, cursor)?;

        Ok(())
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

        for (page, id) in self.pages.iter() {
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
