use crate::{
    memory_management::{
        BytesFormat, ErrorGraph, ManagedMemoryBinding, ManagedMemoryHandle, MemoryLocation,
        MemoryUsage, PageGuard,
        memory_pool::{PageMapping, Slice, calculate_padding},
    },
    server::IoError,
    storage::{StorageHandle, StorageUtilization},
};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::{Debug, Display};
use cubecl_environment::{backtrace::BackTrace, sync::Arc};

/// A memory page is responsible to reserve [slices](Slice) of data based on a fixed [storage buffer](StorageHandle).
pub struct MemoryPage {
    storage: StorageHandle,
    slices: Vec<Slice>,
    /// The free runs the last [merge](Self::merge_free_runs) found, kept to
    /// avoid allocating a new vector all the time.
    free_runs: Vec<FreeRun>,
    /// Memory alignment.
    alignment: u64,
    location_base: MemoryLocation,
    /// Whether the page's [`StorageId`](crate::storage::StorageId) is backed by
    /// a real device allocation. A page carved under a process mode that drops
    /// launches starts unmapped — its id is minted but no driver memory exists
    /// behind it — and is [rebound](Self::rebind_storage) to a real allocation
    /// the first time one of its slices is resolved into a kernel argument,
    /// read or write. Everything else about the page (slice offsets,
    /// coalescing, high-water accounting) behaves identically either way.
    mapped: bool,
    /// Counts the [guards](PageGuard) handed out on the page: it is guarded
    /// while any of them lives.
    guards: Arc<()>,
}

impl MemoryPage {
    /// Creates a new memory page with the given storage and memory alignment.
    ///
    /// `mapping` says whether `storage`'s id is already backed by a device
    /// allocation ([`PageMapping::Eager`]) or is a minted id awaiting one
    /// ([`PageMapping::Lazy`]) — see [`mapped`](Self::mapped).
    pub fn new(
        storage: StorageHandle,
        alignment: u64,
        location_base: MemoryLocation,
        mapping: PageMapping,
    ) -> Self {
        let mut this = MemoryPage {
            storage: storage.clone(),
            slices: Vec::new(),
            free_runs: Vec::new(),
            alignment,
            location_base,
            mapped: matches!(mapping, PageMapping::Eager),
            guards: Arc::new(()),
        };

        let slice = Slice::new(storage, 0);
        let slice_pos = this.slices.len() as u32;
        let mut location = this.location_base;
        location.slice = slice_pos;
        slice.handle.descriptor().update_location(location);
        this.slices.push(slice);

        this
    }

    /// Whether the page's storage id is backed by a real device allocation.
    pub fn is_mapped(&self) -> bool {
        self.mapped
    }

    /// Keep the page as it is for as long as the guard lives.
    pub fn guard(&self) -> PageGuard {
        PageGuard::page(&self.guards)
    }

    /// Whether a [guard](Self::guard) is keeping the page as it is.
    pub fn is_guarded(&self) -> bool {
        Arc::strong_count(&self.guards) > 1
    }

    /// The page's size in bytes.
    pub fn size(&self) -> u64 {
        self.storage.size()
    }

    /// Whether no slice on the page is live.
    pub fn is_empty(&self) -> bool {
        self.slices.iter().all(Slice::is_free)
    }

    /// The slices holding a live allocation.
    pub fn live(&self) -> impl Iterator<Item = &Slice> {
        self.slices.iter().filter(|slice| !slice.is_free())
    }

    /// The slice at `index`, mutably.
    pub fn slice_mut(&mut self, index: usize) -> &mut Slice {
        &mut self.slices[index]
    }

    /// The page-wide storage id (the one every slice on the page shares).
    pub fn storage_id(&self) -> crate::storage::StorageId {
        self.storage.id
    }

    /// Bind the page to a real device allocation: rewrite the page-wide
    /// storage id and every slice's copy of it (offsets and sizes are already
    /// correct — the virtual carving *is* the layout). The old minted id
    /// simply ceases to exist; it never reached the driver.
    pub fn rebind_storage(&mut self, id: crate::storage::StorageId) {
        self.storage.id = id;
        for slice in self.slices.iter_mut() {
            slice.storage.id = id;
        }
        self.mapped = true;
    }

    /// Binds a user defined [`ManagedMemoryHandle`] to a slice in this memory pool.
    pub fn bind(
        &mut self,
        reserved: ManagedMemoryHandle,
        new: ManagedMemoryHandle,
        cursor: u64,
        failures: &mut ErrorGraph,
    ) -> Result<(), IoError> {
        let slice = &mut self.slices[reserved.descriptor().slice()];
        new.descriptor()
            .update_location(reserved.descriptor().location());
        slice.cursor = cursor;
        slice.bind(new, failures);

        Ok(())
    }

    /// Gets the [memory usage](MemoryUsage) of the current memory page.
    pub fn memory_usage(&self) -> MemoryUsage {
        let mut usage = MemoryUsage {
            number_allocs: 0,
            bytes_in_use: 0,
            bytes_padding: 0,
            bytes_reserved: 0,
        };

        for slice in self.slices.iter() {
            usage.bytes_reserved += slice.effective_size();

            if !slice.handle.is_free() {
                usage.number_allocs += 1;
                usage.bytes_in_use += slice.storage.size();
                usage.bytes_padding += slice.padding;
            }
        }

        usage
    }

    /// Gets the [summary](MemoryPageSummary) of the current memory page.
    ///
    /// # Arguments
    ///
    /// - `memory_blocks`: whether the memory block details are included in the summary.
    pub fn summary(&self, memory_blocks: bool) -> MemoryPageSummary {
        let mut summary = MemoryPageSummary::default();

        for slice in self.slices.iter() {
            let is_free = slice.handle.is_free();
            if is_free {
                summary.amount_free += slice.effective_size();
                summary.num_free += 1;
            } else {
                summary.amount_full += slice.effective_size();
                summary.num_full += 1;
            }
            if memory_blocks {
                summary.blocks.push(MemoryBlock {
                    is_free,
                    size: slice.effective_size(),
                });
            }
        }
        summary.amount_total = self.storage.size();
        summary.num_total = self.slices.len();

        summary
    }

    /// Reserves a slice of the given size if there is enough place in the page.
    ///
    /// # Notes
    ///
    /// If the current memory page is fragmented, meaning multiple contiguous slices of data exist,
    /// you can call the [`Self::coalesce()`] function to merge those.
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub fn try_reserve(&mut self, size: u64) -> Option<ManagedMemoryHandle> {
        let effective_size = size + calculate_padding(size, self.alignment);
        let index = self.slices.iter().position(|slice| {
            slice.storage.utilization.size >= effective_size && slice.handle.is_free()
        })?;

        Some(self.reserve_at(index, size))
    }

    /// [Coalesces](Self::coalesce), then reserves a slice of the given size on
    /// the first free slice that fits.
    ///
    /// Whether a slice is free is read once per slice, while merging; the
    /// search that follows only visits the free slices the merge found.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, failures))
    )]
    pub fn reserve(&mut self, size: u64, failures: &mut ErrorGraph) -> Option<ManagedMemoryHandle> {
        self.merge_free_runs(failures);

        let effective_size = size + calculate_padding(size, self.alignment);
        let index = self
            .free_runs
            .iter()
            .map(|run| run.index)
            .find(|&index| self.slices[index].storage.utilization.size >= effective_size)?;

        Some(self.reserve_at(index, size))
    }

    /// Reserves `size` bytes on the free slice at `index`, which fits them,
    /// splitting off what is left as a new free slice.
    fn reserve_at(&mut self, index: usize, size: u64) -> ManagedMemoryHandle {
        let padding = calculate_padding(size, self.alignment);
        let effective_size = size + padding;
        let slice = &mut self.slices[index];

        let can_be_split = slice.storage.utilization.size > effective_size;
        let handle = slice.handle.clone();
        let storage_old = slice.storage.clone();

        // Updates the current storage utilization.
        slice.storage.utilization.size = size;
        slice.padding = padding;

        if can_be_split {
            let new_slice = Slice::new(storage_old.offset_start(effective_size), 0);
            self.insert_slice(index + 1, new_slice);
        }

        handle
    }

    /// Gets the [storage handle](SliceHandle) with the correct offset and size using the slice
    /// binding.
    ///
    /// If the handle isn't returned, it means the binding isn't present in the given page.
    pub fn find(&self, binding: &ManagedMemoryBinding) -> Result<&Slice, IoError> {
        let slice_index = binding.descriptor().slice();

        self.slices
            .get(slice_index)
            .ok_or_else(|| IoError::NotFound {
                backtrace: BackTrace::capture(),
                reason: alloc::format!("Memory slice {} doesn't exist", slice_index).into(),
            })
    }

    /// [`find`](Self::find), mutably.
    pub fn find_mut(&mut self, binding: &ManagedMemoryBinding) -> Result<&mut Slice, IoError> {
        let slice_index = binding.descriptor().slice();

        self.slices
            .get_mut(slice_index)
            .ok_or_else(|| IoError::NotFound {
                backtrace: BackTrace::capture(),
                reason: alloc::format!("Memory slice {} doesn't exist", slice_index).into(),
            })
    }

    /// Release every failure the page's slices still carry, for a page about
    /// to be dropped whole: the slices go with it, and a tag on one must not
    /// outlive its carrier.
    pub fn shed(&mut self, failures: &mut ErrorGraph) {
        for slice in self.slices.iter_mut() {
            slice.tainted.clear(failures);
        }
    }

    pub fn update_page(&mut self, page: u16) {
        self.location_base.page = page;

        for slice in self.slices.iter() {
            slice.descriptor().update_page(page);
        }
    }

    /// Recompute the memory page metadata to make sure adjacent slices are merged together into a
    /// single slice.
    ///
    /// This is necessary to allow bigger slices to be reserved on the current page.
    ///
    /// A merged slice is dropped here, so the failure it carried is released
    /// on the spot. Free slices are the only ones merged, and nobody holds a
    /// handle to a free slice, so losing the tag costs nothing but the
    /// decrement.
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip_all))]
    pub fn coalesce(&mut self, failures: &mut ErrorGraph) {
        self.merge_free_runs(failures);
    }

    /// Merges every run of adjacent free slices into one, in place, and
    /// records each run with the index of the slice it became in
    /// [`free_runs`](Self::free_runs).
    ///
    /// Only the slices that move are renumbered: those before the first
    /// merged run keep their index.
    fn merge_free_runs(&mut self, failures: &mut ErrorGraph) {
        self.free_runs.clear();
        let mut current: Option<FreeRun> = None;
        for (index, slice) in self.slices.iter().enumerate() {
            if slice.is_free() {
                let run = current.get_or_insert(FreeRun {
                    start: index,
                    count: 0,
                    size: 0,
                    index,
                });
                run.count += 1;
                run.size += slice.effective_size();
            } else if let Some(run) = current.take() {
                self.free_runs.push(run);
            }
        }
        self.free_runs.extend(current);

        if self.free_runs.iter().all(|run| run.count == 1) {
            return;
        }

        let storage = &self.storage;
        let location_base = self.location_base;
        let mut runs = self.free_runs.iter_mut().peekable();
        let mut index = 0;
        let mut kept = 0;

        self.slices.retain_mut(|slice| {
            let current = index;
            index += 1;

            let keep = match runs.peek_mut() {
                Some(run) if run.count > 1 && current >= run.start => {
                    slice.tainted.clear(failures);
                    let first = current == run.start;
                    if first {
                        let mut merged = storage.clone();
                        merged.utilization = StorageUtilization {
                            offset: slice.storage.utilization.offset,
                            size: run.size,
                        };
                        *slice = Slice::new(merged, 0);
                        let mut location = location_base;
                        location.slice = kept as u32;
                        slice.descriptor().update_location(location);
                        run.index = kept;
                    }
                    if current + 1 == run.start + run.count {
                        runs.next();
                    }
                    first
                }
                run => {
                    if let Some(run) = run.filter(|run| run.start == current) {
                        run.index = kept;
                        runs.next();
                    }
                    if kept != current {
                        slice.descriptor().update_slice(kept as u32);
                    }
                    true
                }
            };

            if keep {
                kept += 1;
            }
            keep
        });
    }

    /// Inserts `slice` at `index`, renumbering the slices after it.
    fn insert_slice(&mut self, index: usize, slice: Slice) {
        let mut location = self.location_base;
        location.slice = index as u32;
        slice.descriptor().update_location(location);
        self.slices.insert(index, slice);

        for (position, slice) in self.slices.iter().enumerate().skip(index + 1) {
            slice.descriptor().update_slice(position as u32);
        }
    }
}

/// A run of adjacent free slices on a [`MemoryPage`].
#[derive(Debug, Clone, Copy)]
struct FreeRun {
    /// The index of its first slice, before merging.
    start: usize,
    /// How many slices it spans.
    count: usize,
    /// The bytes it spans, padding included.
    size: u64,
    /// The index of the slice it is, once merged.
    index: usize,
}

#[derive(Debug, PartialEq, Eq)]
struct MemoryBlock {
    is_free: bool,
    size: u64,
}

#[derive(Default, PartialEq, Eq)]
pub struct MemoryPageSummary {
    blocks: Vec<MemoryBlock>,
    pub amount_free: u64,
    pub amount_full: u64,
    pub amount_total: u64,
    pub num_free: usize,
    pub num_full: usize,
    pub num_total: usize,
}

impl Display for MemoryBlock {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.is_free {
            true => f.write_fmt(format_args!("Free ({})", BytesFormat::new(self.size))),
            false => f.write_fmt(format_args!("Reserved ({})", BytesFormat::new(self.size))),
        }
    }
}
impl Display for MemoryPageSummary {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{self:?}"))
    }
}

impl Debug for MemoryPageSummary {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("\n==== Memory Page Summary ====\n")?;
        f.write_str("[Info]\n")?;

        for (tag, num, amount) in [
            ("Free ", self.num_free, self.amount_free),
            ("Full ", self.num_full, self.amount_full),
            ("Total", self.num_total, self.amount_total),
        ] {
            f.write_fmt(format_args!(
                " - {tag}: {} slices ({})\n",
                num,
                BytesFormat::new(amount),
            ))?;
        }

        f.write_str("\n[Blocks]\n")?;
        let mut blocks = String::new();
        for (i, b) in self.blocks.iter().enumerate() {
            if i == 0 {
                blocks += "|";
            }
            blocks += format!(" {b} |").as_str();
        }
        let size = blocks.len();
        for _ in 0..size {
            f.write_str("-")?;
        }
        f.write_str("\n")?;
        f.write_str(&blocks)?;
        f.write_str("\n")?;
        for _ in 0..size {
            f.write_str("-")?;
        }

        f.write_str("\n=============================")?;
        f.write_str("\n")
    }
}

impl Display for MemoryPage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{}", self.summary(true)))
    }
}

#[cfg(test)]
#[allow(clippy::bool_assert_comparison, clippy::identity_op)]
mod tests {
    use crate::storage::{StorageId, StorageUtilization};
    use alloc::vec;

    use super::*;

    const MB: u64 = 1024 * 1024;

    #[test_log::test]
    fn test_memory_page() {
        let mut page = new_memory_page(32 * MB);
        let slice = page
            .try_reserve(16 * MB)
            .expect("Enough space to allocate a new slice");

        assert_eq!(slice.is_free(), false);
        assert_eq!(slice.can_mut(), true);

        let storage = &page
            .find(&slice.binding())
            .expect("To find the correct storage")
            .storage;

        assert_eq!(
            storage.utilization,
            StorageUtilization {
                offset: 0,
                size: 16 * MB
            },
            "Utilization to be correct"
        );

        let summary = page.summary(true);

        assert_eq!(
            summary,
            MemoryPageSummary {
                blocks: vec![
                    MemoryBlock {
                        is_free: true,
                        size: 16 * MB
                    },
                    MemoryBlock {
                        is_free: true,
                        size: 16 * MB
                    }
                ],
                amount_free: 32 * MB,
                amount_full: 0,
                amount_total: 32 * MB,
                num_free: 2,
                num_full: 0,
                num_total: 2
            },
            "Summary is correct before coalesce",
        );
        page.coalesce(&mut ErrorGraph::default());
        let summary = page.summary(true);

        assert_eq!(
            summary,
            MemoryPageSummary {
                blocks: vec![MemoryBlock {
                    is_free: true,
                    size: 32 * MB
                },],
                amount_free: 32 * MB,
                amount_full: 0,
                amount_total: 32 * MB,
                num_free: 1,
                num_full: 0,
                num_total: 1
            },
            "Summary is correct after coalesce",
        );
    }

    #[test_log::test]
    fn test_reserve_merges_and_splits_in_place() {
        let mut page = new_memory_page(32 * MB);
        let mut failures = ErrorGraph::default();
        let [h0, h1, h2, h3, h4, h5, h6, h7] =
            core::array::from_fn(|_| page.try_reserve(2 * MB).expect("Enough space"));

        // Free a run of two, a lone slice, and a run that reaches the free tail.
        core::mem::drop((h1, h2, h4, h6, h7));

        // Merging the first run moves every slice after it back one index, and
        // splitting it moves them forward again: each handle must follow.
        let first = page
            .reserve(3 * MB, &mut failures)
            .expect("The 4 MB run fits");
        // Too large for the run's remainder and the lone slice.
        let tail = page.reserve(6 * MB, &mut failures).expect("The tail fits");

        for (handle, offset) in [(&h0, 0), (&first, 2), (&h3, 6), (&h5, 10), (&tail, 12)] {
            let slice = page.find(&handle.clone().binding()).unwrap();
            assert_eq!(slice.handle.descriptor(), handle.descriptor());
            assert_eq!(slice.storage.utilization.offset, offset * MB);
        }
    }

    #[test_log::test]
    fn test_scenario() {
        let mut page = new_memory_page(32 * MB);

        let slice_1 = page
            .try_reserve(4 * MB)
            .expect("Enough space to allocate a new slice");
        let slice_2 = page
            .try_reserve(15 * MB)
            .expect("Enough space to allocate a new slice");
        let slice_3 = page
            .try_reserve(8 * MB)
            .expect("Enough space to allocate a new slice");
        let slice_4 = page
            .try_reserve(4 * MB)
            .expect("Enough space to allocate a new slice");

        assert_eq!(
            page.summary(true),
            MemoryPageSummary {
                blocks: vec![
                    MemoryBlock {
                        is_free: false,
                        size: 4 * MB
                    },
                    MemoryBlock {
                        is_free: false,
                        size: 15 * MB
                    },
                    MemoryBlock {
                        is_free: false,
                        size: 8 * MB
                    },
                    MemoryBlock {
                        is_free: false,
                        size: 4 * MB
                    },
                    MemoryBlock {
                        is_free: true,
                        size: 1 * MB
                    }
                ],
                amount_free: 1 * MB,
                amount_full: 31 * MB,
                amount_total: 32 * MB,
                num_free: 1,
                num_full: 4,
                num_total: 5
            },
        );

        let slice_5 = page.try_reserve(8 * MB);
        assert!(slice_5.is_none(), "No more place");

        core::mem::drop(slice_2);
        let slice_5 = page.try_reserve(9 * MB);
        assert!(slice_5.is_some(), "Now we have more place");

        let slice_6 = page.try_reserve(9 * MB);
        assert!(slice_6.is_none(), "No more place");

        core::mem::drop(slice_3);
        let slice_6 = page.try_reserve(9 * MB);
        assert!(slice_6.is_none(), "No more place");

        page.coalesce(&mut ErrorGraph::default());

        assert_eq!(
            page.summary(true),
            MemoryPageSummary {
                blocks: vec![
                    MemoryBlock {
                        is_free: false,
                        size: 4 * MB
                    },
                    MemoryBlock {
                        is_free: false,
                        size: 9 * MB
                    },
                    MemoryBlock {
                        is_free: true,
                        size: 14 * MB
                    },
                    MemoryBlock {
                        is_free: false,
                        size: 4 * MB
                    },
                    MemoryBlock {
                        is_free: true,
                        size: 1 * MB
                    }
                ],
                amount_free: 15 * MB,
                amount_full: 17 * MB,
                amount_total: 32 * MB,
                num_free: 2,
                num_full: 3,
                num_total: 5
            },
        );

        assert_eq!(
            page.find(&slice_4.clone().binding())
                .unwrap()
                .storage
                .utilization,
            StorageUtilization {
                offset: 27 * MB,
                size: 4 * MB
            },
            "Utilization to be correct"
        );

        let slice_6 = page.try_reserve(9 * MB);
        assert!(slice_6.is_some(), "Now we have more place");
        core::mem::drop(slice_1);
        core::mem::drop(slice_4);

        page.coalesce(&mut ErrorGraph::default());

        assert_eq!(
            page.find(&slice_6.clone().unwrap().binding())
                .unwrap()
                .storage
                .utilization,
            StorageUtilization {
                offset: 13 * MB,
                size: 9 * MB
            },
            "Utilization to be correct"
        );

        assert_eq!(
            page.summary(true),
            MemoryPageSummary {
                blocks: vec![
                    MemoryBlock {
                        is_free: true,
                        size: 4 * MB
                    },
                    MemoryBlock {
                        is_free: false,
                        size: 9 * MB
                    },
                    MemoryBlock {
                        is_free: false,
                        size: 9 * MB
                    },
                    MemoryBlock {
                        is_free: true,
                        size: 10 * MB
                    }
                ],
                amount_free: 14 * MB,
                amount_full: 18 * MB,
                amount_total: 32 * MB,
                num_free: 2,
                num_full: 2,
                num_total: 4
            },
        );
    }

    fn new_memory_page(size: u64) -> MemoryPage {
        let storage = StorageHandle::new(StorageId::new(), StorageUtilization { offset: 0, size });

        MemoryPage::new(storage, 4, MemoryLocation::new(0, 0, 0), PageMapping::Eager)
    }
}
