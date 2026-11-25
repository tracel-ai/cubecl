use crate::{
    memory_management::{
        BytesFormat, ErrorGraph, ManagedMemoryBinding, ManagedMemoryHandle, MemoryLocation,
        MemoryUsage, PageGuard,
        memory_pool::{
            PageMapping, Slice, SliceAllocator, SliceBinder, missing, padded_size, slices_usage,
        },
    },
    server::IoError,
    storage::{StorageAllocator, StorageHandle, StorageId, StorageUtilization},
};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::{Debug, Display};
use cubecl_environment::sync::Arc;

/// Suballocates one device page into [slices](Slice) of a fixed
/// [storage buffer](StorageHandle): the allocations on it and the free room
/// between them.
///
/// # Assumptions
///
/// `failures`, in every function that takes it, is the [`ErrorGraph`] that
/// holds the failures the page's slices carry.
pub(crate) struct PageSuballocator {
    storage: StorageHandle,
    /// The slices in offset order, read by the position a location carries
    /// (Position) and scanned in offset order for room (Ordered).
    slices: Vec<Slice>,
    alignment: u64,
    location_base: MemoryLocation,
    /// Whether the page's [`StorageId`] is backed by a real device
    /// allocation. A page carved under a process mode that drops launches
    /// starts unmapped — its id is minted but no driver memory exists behind
    /// it — and is [rebound](Self::rebind_storage) to a real allocation the
    /// first time one of its slices is resolved into a kernel argument, read
    /// or write. Everything else about the page (slice offsets, coalescing,
    /// high-water accounting) behaves identically either way.
    mapped: bool,
    /// Counts the [guards](PageGuard) handed out on the page: it is guarded
    /// while any of them lives.
    guards: Arc<()>,
}

/// What a [`PageSuballocator`] holds, as its readers see it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PageView<'a> {
    /// The page's slices, read by the position a location carries (Position)
    /// and in offset order (Ordered).
    pub slices: &'a [Slice],
    /// Whether the page's storage id is backed by a real device allocation.
    pub is_mapped: bool,
    /// Whether a guard is keeping the page as it is.
    pub is_guarded: bool,
    /// The page's size.
    pub size_bytes: u64,
    /// The page-wide storage id, the one every slice on the page shares.
    pub storage_id: StorageId,
}

impl PageSuballocator {
    /// A page of one free slice over `storage`, whose slices start at
    /// `alignment`, backed now or later as `mapping` says.
    pub(crate) fn new(
        storage: StorageHandle,
        alignment: u64,
        location_base: MemoryLocation,
        mapping: PageMapping,
    ) -> Self {
        let slice = Slice::new(storage.clone(), 0);
        slice.descriptor().update_location(location_base);
        Self {
            storage,
            slices: alloc::vec![slice],
            alignment,
            location_base,
            mapped: mapping == PageMapping::Eager,
            guards: Arc::new(()),
        }
    }

    /// What the page holds right now.
    pub(crate) fn view(&self) -> PageView<'_> {
        PageView {
            slices: &self.slices,
            is_mapped: self.mapped,
            is_guarded: Arc::strong_count(&self.guards) > 1,
            size_bytes: self.storage.size(),
            storage_id: self.storage.id,
        }
    }

    /// The slice `binding`'s location names.
    ///
    /// # Errors
    ///
    /// [`IoError::NotFound`] when the page holds no slice there.
    pub(crate) fn find(&self, binding: &ManagedMemoryBinding) -> Result<&Slice, IoError> {
        let index = binding.descriptor().slice();
        self.slices
            .get(index)
            .ok_or_else(|| missing("slice", index))
    }

    /// The slice `binding`'s location names, mutably.
    ///
    /// # Errors
    ///
    /// [`IoError::NotFound`] when the page holds no slice there.
    pub(crate) fn find_mut(
        &mut self,
        binding: &ManagedMemoryBinding,
    ) -> Result<&mut Slice, IoError> {
        let index = binding.descriptor().slice();
        self.slices
            .get_mut(index)
            .ok_or_else(|| missing("slice", index))
    }

    /// The slice at `index`, mutably.
    ///
    /// # Panics
    ///
    /// When the page holds no slice at `index`.
    pub(crate) fn slice_at_index_mut(&mut self, index: usize) -> &mut Slice {
        &mut self.slices[index]
    }

    /// Keep the page as it is for as long as the guard lives.
    pub(crate) fn guard(&mut self) -> PageGuard {
        PageGuard::page(&self.guards)
    }

    /// Bind the page to the real device allocation `id`: the page and every
    /// slice take it (offsets and sizes are already correct — the virtual
    /// carving *is* the layout). The old minted id simply ceases to exist; it
    /// never reached the driver.
    pub(crate) fn rebind_storage(&mut self, id: StorageId) {
        for slice in self.slices.iter_mut() {
            slice.storage.id = id;
        }
        self.storage.id = id;
        self.mapped = true;
    }

    /// Bind `assigned` to the slice `reserved` holds, at `cursor`.
    ///
    /// # Panics
    ///
    /// When `reserved` names no slice of the page.
    pub(crate) fn bind(
        &mut self,
        reserved: ManagedMemoryHandle,
        assigned: ManagedMemoryHandle,
        cursor: u64,
        failures: &mut ErrorGraph,
    ) {
        let slice = self
            .slices
            .get_mut(reserved.descriptor().slice())
            .expect("a reserved slice should be on the page that reserved it");
        SliceBinder.bind(slice, assigned, cursor, failures);
    }

    /// Reserve `size_bytes` on the first free slice in offset order that
    /// fits them, padded, splitting off the room it leaves. `None` when no
    /// free slice fits them: [`coalesce`](Self::coalesce) merges neighbouring
    /// free slices into larger ones.
    pub(crate) fn suballocate_bytes(&mut self, size_bytes: u64) -> Option<ManagedMemoryHandle> {
        let effective_size = padded_size(size_bytes, self.alignment)?;
        let index = self
            .slices
            .iter()
            .position(|slice| slice.is_free() && slice.effective_size() >= effective_size)?;
        let slice = &mut self.slices[index];
        let rest = Self::split(slice, size_bytes, effective_size);
        let handle = slice.handle.clone();
        if let Some(rest) = rest {
            self.insert_after(index, rest);
        }
        Some(handle)
    }

    /// Number the page `page_index` in every location its slices carry.
    pub(crate) fn set_page_index(&mut self, page_index: u16) {
        self.location_base.page = page_index;
        self.number_slices();
    }

    /// Merge every run of neighbouring free slices into one, so larger
    /// allocations fit the page again. A merged slice is dropped here, so the
    /// failure it carried is released on the spot; nobody holds a handle to a
    /// free slice, so losing the tag costs nothing but the decrement.
    pub(crate) fn coalesce(&mut self, failures: &mut ErrorGraph) {
        self.slices
            .dedup_by(|later, earlier| Self::merge_free(later, earlier, failures));
        self.number_slices();
    }

    /// Give the page real device backing from `storage`, if it was carved
    /// lazily, and [rebind](Self::rebind_storage) it to that allocation.
    ///
    /// # Errors
    ///
    /// [`IoError::StorageMappingFailed`] when the storage refuses the backing.
    pub(crate) fn map<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
    ) -> Result<(), IoError> {
        if self.mapped {
            return Ok(());
        }
        let real = SliceAllocator::new(storage).allocate_backing(self.storage.size())?;
        self.rebind_storage(real.id);
        Ok(())
    }

    /// Give the page back to `storage` when, once coalesced, nothing is live
    /// on it and nothing guards it, answering whether it went. The failures
    /// its slices still carry are released, since a tag must not outlive its
    /// carrier; an unmapped page has nothing behind its minted id, so the
    /// driver is handed nothing for it.
    pub(crate) fn release_if_unused<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        failures: &mut ErrorGraph,
    ) -> bool {
        self.coalesce(failures);
        if !self.view().is_empty() || self.view().is_guarded {
            return false;
        }
        self.release(failures);
        if self.mapped {
            storage.deallocate(self.storage.id);
        }
        true
    }

    /// Release every failure the page's slices still carry, for a page about
    /// to be dropped whole.
    fn release(&mut self, failures: &mut ErrorGraph) {
        for slice in self.slices.iter_mut() {
            slice.tainted.clear(failures);
        }
    }

    /// Give every slice the location of the page with its index.
    fn number_slices(&self) {
        for (index, slice) in self.slices.iter().enumerate() {
            slice.descriptor().update_location(MemoryLocation {
                slice: index as u32,
                ..self.location_base
            });
        }
    }

    /// Place `slice` right after the slice at `index`. Only the slices after
    /// it move, so only their descriptors learn a new position: a reservation
    /// splits a slice on every launch that allocates, and a page holds many.
    fn insert_after(&mut self, index: usize, slice: Slice) {
        self.slices.insert(index + 1, slice);
        for (position, slice) in self.slices.iter().enumerate().skip(index + 1) {
            slice.descriptor().update_location(MemoryLocation {
                slice: position as u32,
                ..self.location_base
            });
        }
    }

    /// Serve `size` bytes, `effective_size` once padded, from the start of
    /// the free `slice`, whose room is its padded size: the padding its last
    /// allocation took is part of it. Answers the slice of the room left
    /// after them, if any is.
    fn split(slice: &mut Slice, size: u64, effective_size: u64) -> Option<Slice> {
        let capacity = slice.effective_size();
        let mut rest = slice.storage.clone();
        rest.utilization = StorageUtilization {
            offset_bytes: rest.offset() + effective_size,
            size_bytes: capacity - effective_size,
        };
        slice.storage.utilization.size_bytes = size;
        slice.padding_bytes = effective_size - size;
        (capacity > effective_size).then(|| Slice::new(rest, 0))
    }

    /// Merge `later` into `earlier` when both are free, answering whether it
    /// did. The merged slice is a fresh one over both, and both release the
    /// failures they carried.
    fn merge_free(later: &mut Slice, earlier: &mut Slice, failures: &mut ErrorGraph) -> bool {
        if !(later.is_free() && earlier.is_free()) {
            return false;
        }
        later.tainted.clear(failures);
        earlier.tainted.clear(failures);
        let mut storage = earlier.storage.clone();
        storage.utilization.size_bytes = earlier.effective_size() + later.effective_size();
        *earlier = Slice::new(storage, 0);
        true
    }
}

impl<'a> PageView<'a> {
    /// Whether no allocation on the page is live.
    pub(crate) fn is_empty(&self) -> bool {
        self.slices.iter().all(Slice::is_free)
    }

    /// The slices holding a live allocation, in offset order.
    pub(crate) fn live(&self) -> impl Iterator<Item = &'a Slice> + 'a {
        self.slices.iter().filter(|slice| !slice.is_free())
    }

    /// The page's usage: its live allocations and the bytes it holds.
    pub(crate) fn usage(&self) -> MemoryUsage {
        slices_usage(self.slices.iter())
    }

    /// The storage behind the slice at `index`, while the page has real
    /// device backing: `None` for a page carved lazily and never resolved,
    /// which has nothing behind it to read.
    ///
    /// # Panics
    ///
    /// When the page holds no slice at `index`.
    pub(crate) fn mapped_storage_at_index(&self, index: usize) -> Option<StorageHandle> {
        self.is_mapped.then(|| self.slices[index].storage.clone())
    }

    /// The page's free and full room, and each slice as a block when
    /// `with_blocks` asks for them.
    pub(crate) fn summary_with_blocks(&self, with_blocks: bool) -> MemoryPageSummary {
        MemoryPageSummary {
            blocks: self.blocks(with_blocks),
            free_bytes: self.free().map(Slice::effective_size).sum(),
            full_bytes: self.live().map(Slice::effective_size).sum(),
            total_bytes: self.size_bytes,
            free_count: self.free().count(),
            full_count: self.live().count(),
            total_count: self.slices.len(),
        }
    }

    /// The free slices, in offset order.
    fn free(&self) -> impl Iterator<Item = &'a Slice> + 'a {
        self.slices.iter().filter(|slice| slice.is_free())
    }

    /// Each slice as a block, when `with_blocks` asks for them.
    fn blocks(&self, with_blocks: bool) -> Vec<MemoryBlock> {
        if !with_blocks {
            return Vec::new();
        }
        self.slices
            .iter()
            .map(|slice| MemoryBlock {
                is_free: slice.is_free(),
                size_bytes: slice.effective_size(),
            })
            .collect()
    }
}

/// One slice of a page, as a summary lists it.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct MemoryBlock {
    /// Whether no allocation is live on it.
    pub is_free: bool,
    /// Its padded size.
    pub size_bytes: u64,
}

/// A page's free and full room.
#[derive(Default, PartialEq, Eq)]
pub(crate) struct MemoryPageSummary {
    /// The page's slices, in offset order as they were listed (Arrival),
    /// when asked for.
    pub blocks: Vec<MemoryBlock>,
    /// The bytes free slices hold.
    pub free_bytes: u64,
    /// The bytes live slices hold.
    pub full_bytes: u64,
    /// The page's size.
    pub total_bytes: u64,
    /// The free slices.
    pub free_count: usize,
    /// The live slices.
    pub full_count: usize,
    /// Every slice.
    pub total_count: usize,
}

impl Display for MemoryBlock {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let size = BytesFormat::from_byte_count(self.size_bytes);
        if self.is_free {
            return write!(f, "Free ({size})");
        }
        write!(f, "Reserved ({size})")
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

        for (tag, count, bytes) in [
            ("Free ", self.free_count, self.free_bytes),
            ("Full ", self.full_count, self.full_bytes),
            ("Total", self.total_count, self.total_bytes),
        ] {
            f.write_fmt(format_args!(
                " - {tag}: {count} slices ({})\n",
                BytesFormat::from_byte_count(bytes),
            ))?;
        }

        f.write_str("\n[Blocks]\n")?;
        let blocks: String = self
            .blocks
            .iter()
            .map(|block| format!(" {block} |"))
            .collect();
        let blocks = format!("|{blocks}");
        let rule = "-".repeat(blocks.len());
        f.write_fmt(format_args!("{rule}\n{blocks}\n{rule}"))?;

        f.write_str("\n=============================")?;
        f.write_str("\n")
    }
}

impl Display for PageSuballocator {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{}", self.view().summary_with_blocks(true)))
    }
}

#[cfg(test)]
#[allow(clippy::identity_op)]
mod tests {
    use super::*;
    use crate::memory_management::memory_pool::CountingStorageAllocator;
    use crate::storage::StorageAllocator;
    use alloc::vec;

    const MB: u64 = 1024 * 1024;

    fn block(is_free: bool, size_bytes: u64) -> MemoryBlock {
        MemoryBlock {
            is_free,
            size_bytes,
        }
    }

    fn reserve(page: &mut PageSuballocator, size_bytes: u64) -> ManagedMemoryHandle {
        page.suballocate_bytes(size_bytes)
            .expect("the page should have room for the slice")
    }

    fn offset_of(page: &PageSuballocator, handle: &ManagedMemoryHandle) -> StorageUtilization {
        page.find(&handle.clone().binding())
            .expect("the slice should be on the page")
            .storage
            .utilization
            .clone()
    }

    /// A reservation splits the free slice it lands on, its storage starting
    /// where that slice did.
    #[test_log::test]
    fn a_reservation_splits_the_slice_it_lands_on() {
        let mut page = new_memory_page(32 * MB);
        let slice = reserve(&mut page, 16 * MB);

        assert_eq!(
            offset_of(&page, &slice),
            StorageUtilization {
                offset_bytes: 0,
                size_bytes: 16 * MB
            },
        );
        assert_eq!(page.view().summary_with_blocks(true).total_count, 2);
    }

    /// Coalescing merges the free slices a freed reservation leaves back into
    /// one.
    #[test_log::test]
    fn coalescing_merges_neighbouring_free_slices() {
        let mut page = new_memory_page(32 * MB);
        core::mem::drop(reserve(&mut page, 16 * MB));

        page.coalesce(&mut ErrorGraph::default());

        assert_eq!(
            page.view().summary_with_blocks(true),
            MemoryPageSummary {
                blocks: vec![block(true, 32 * MB)],
                free_bytes: 32 * MB,
                full_bytes: 0,
                total_bytes: 32 * MB,
                free_count: 1,
                full_count: 0,
                total_count: 1
            },
        );
    }

    /// Reservations fill a page in offset order.
    #[test_log::test]
    fn reservations_fill_a_page_in_order() {
        let mut page = new_memory_page(32 * MB);
        let _slices = [4, 15, 8, 4].map(|size| reserve(&mut page, size * MB));

        assert_eq!(
            page.view().summary_with_blocks(true),
            MemoryPageSummary {
                blocks: vec![
                    block(false, 4 * MB),
                    block(false, 15 * MB),
                    block(false, 8 * MB),
                    block(false, 4 * MB),
                    block(true, 1 * MB)
                ],
                free_bytes: 1 * MB,
                full_bytes: 31 * MB,
                total_bytes: 32 * MB,
                free_count: 1,
                full_count: 4,
                total_count: 5
            },
        );
    }

    /// A reservation no free slice fits is refused.
    #[test_log::test]
    fn a_reservation_no_free_slice_fits_is_refused() {
        let mut page = new_memory_page(32 * MB);
        let _slices = [4, 15, 8, 4].map(|size| reserve(&mut page, size * MB));

        assert!(page.suballocate_bytes(8 * MB).is_none());
    }

    /// A summary asked without blocks lists none.
    #[test_log::test]
    fn a_summary_without_blocks_lists_none() {
        let page = new_memory_page(32 * MB);

        assert!(page.view().summary_with_blocks(false).blocks.is_empty());
    }

    /// Two neighbouring free slices serve only what each fits apart.
    #[test_log::test]
    fn neighbouring_free_slices_serve_apart_until_coalesced() {
        let mut page = new_memory_page(32 * MB);
        let [_first, second, third, _fourth] =
            [4, 15, 8, 4].map(|size| reserve(&mut page, size * MB));
        core::mem::drop(second);
        let _nine = reserve(&mut page, 9 * MB);
        core::mem::drop(third);

        assert!(page.suballocate_bytes(9 * MB).is_none(), "6 and 8 MB apart");
    }

    /// Coalesced, neighbouring free slices serve what they fit together.
    #[test_log::test]
    fn coalesced_free_slices_serve_together() {
        let mut page = new_memory_page(32 * MB);
        let [_first, second, third, _fourth] =
            [4, 15, 8, 4].map(|size| reserve(&mut page, size * MB));
        core::mem::drop(second);
        let _nine = reserve(&mut page, 9 * MB);
        core::mem::drop(third);

        page.coalesce(&mut ErrorGraph::default());

        assert!(page.suballocate_bytes(9 * MB).is_some(), "14 MB together");
    }

    /// Coalescing keeps the live slices' offsets.
    #[test_log::test]
    fn coalescing_keeps_live_offsets() {
        let mut page = new_memory_page(32 * MB);
        let [_first, second, third, fourth] =
            [4, 15, 8, 4].map(|size| reserve(&mut page, size * MB));
        core::mem::drop((second, third));

        page.coalesce(&mut ErrorGraph::default());

        assert_eq!(
            offset_of(&page, &fourth),
            StorageUtilization {
                offset_bytes: 27 * MB,
                size_bytes: 4 * MB
            },
        );
    }

    /// A freed slice's room includes the padding its last allocation took, so
    /// the same allocation fits it again.
    #[test_log::test]
    fn a_freed_slice_serves_its_padded_size_again() {
        let mut page = aligned_memory_page(128, 32);
        core::mem::drop(reserve(&mut page, 100));

        assert!(
            page.suballocate_bytes(100).is_some(),
            "the slice is 128 bytes"
        );
    }

    /// Splitting a freed slice leaves the rest of its padded room on the page.
    #[test_log::test]
    fn splitting_a_freed_slice_keeps_its_padding_on_the_page() {
        let mut page = aligned_memory_page(128, 32);
        core::mem::drop(reserve(&mut page, 100));

        let _split = reserve(&mut page, 32);

        assert_eq!(page.view().usage().reserved_bytes, 128);
    }

    /// A size whose padding passes the largest `u64` fits no slice.
    #[test_log::test]
    fn a_size_whose_padding_overflows_fits_no_slice() {
        let mut page = aligned_memory_page(128, 32);

        assert!(page.suballocate_bytes(u64::MAX).is_none());
    }

    /// A binding resolves to the slice it was bound to.
    #[test_log::test]
    fn a_binding_resolves_to_its_slice() {
        let mut page = new_memory_page(32 * MB);
        let reserved = reserve(&mut page, MB);
        let assigned = ManagedMemoryHandle::new();

        page.bind(reserved, assigned.clone(), 3, &mut ErrorGraph::default());

        assert!(page.find(&assigned.binding()).is_ok());
    }

    /// A lazily carved page reads no storage.
    #[test_log::test]
    fn a_lazy_page_has_no_storage() {
        let page = lazy_page();

        assert!(page.view().mapped_storage_at_index(0).is_none());
    }

    /// A rebound page reads the storage it was rebound to.
    #[test_log::test]
    fn a_rebound_page_reads_its_new_storage() {
        let mut page = lazy_page();
        let real = StorageId::new();

        page.rebind_storage(real);

        let mapped = page
            .view()
            .mapped_storage_at_index(0)
            .expect("a rebound page should have storage");
        assert_eq!(mapped.id, real);
    }

    /// A lazy page is mapped from the storage it is given.
    #[test_log::test]
    fn a_lazy_page_is_mapped_from_the_storage() {
        let mut storage = CountingStorageAllocator::default();
        let mut page = lazy_page();

        page.map(&mut storage)
            .expect("the host should back the page");

        assert!(page.view().is_mapped);
        assert_eq!(storage.bytes_allocated(), MB);
    }

    /// An unused page goes back to the driver.
    #[test_log::test]
    fn an_unused_page_is_released() {
        let mut storage = CountingStorageAllocator::default();
        let mut page = new_memory_page(MB);
        let id = page.view().storage_id;

        assert!(page.release_if_unused(&mut storage, &mut ErrorGraph::default()));
        assert_eq!(storage.into_deallocated(), [id]);
    }

    /// An unused page never backed hands the driver nothing.
    #[test_log::test]
    fn an_unused_unmapped_page_hands_back_nothing() {
        let mut storage = CountingStorageAllocator::default();
        let mut page = lazy_page();

        assert!(page.release_if_unused(&mut storage, &mut ErrorGraph::default()));
        assert!(storage.into_deallocated().is_empty());
    }

    /// A page with a live allocation stays.
    #[test_log::test]
    fn a_page_in_use_stays() {
        let mut storage = CountingStorageAllocator::default();
        let mut page = new_memory_page(MB);
        let _live = reserve(&mut page, 64);

        assert!(!page.release_if_unused(&mut storage, &mut ErrorGraph::default()));
    }

    /// A guarded page stays, empty or not.
    #[test_log::test]
    fn a_guarded_page_stays() {
        let mut storage = CountingStorageAllocator::default();
        let mut page = new_memory_page(MB);
        let _guard = page.guard();

        assert!(!page.release_if_unused(&mut storage, &mut ErrorGraph::default()));
    }

    fn lazy_page() -> PageSuballocator {
        let storage = StorageHandle::new(
            StorageId::new(),
            StorageUtilization {
                offset_bytes: 0,
                size_bytes: MB,
            },
        );
        PageSuballocator::new(storage, 4, MemoryLocation::new(0, 0, 0), PageMapping::Lazy)
    }

    fn new_memory_page(size: u64) -> PageSuballocator {
        aligned_memory_page(size, 4)
    }

    fn aligned_memory_page(size: u64, alignment: u64) -> PageSuballocator {
        let storage = StorageHandle::new(
            StorageId::new(),
            StorageUtilization {
                offset_bytes: 0,
                size_bytes: size,
            },
        );

        PageSuballocator::new(
            storage,
            alignment,
            MemoryLocation::new(0, 0, 0),
            PageMapping::Eager,
        )
    }

    /// Mapping a page already backed allocates nothing.
    #[test_log::test]
    fn mapping_a_backed_page_allocates_nothing() {
        let mut storage = CountingStorageAllocator::default();
        let mut page = new_memory_page(MB);

        page.map(&mut storage)
            .expect("a backed page should need nothing");

        assert_eq!(storage.bytes_allocated(), 0);
    }
}
