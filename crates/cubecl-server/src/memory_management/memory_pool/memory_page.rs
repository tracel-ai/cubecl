use crate::{
    memory_management::{
        BytesFormat, ErrorGraph, ManagedMemoryBinding, ManagedMemoryHandle, MemoryLocation,
        MemoryUsage, PageGuard,
        memory_pool::{PageMapping, Slice, bind_slice, missing, padded_size, slices_usage},
    },
    server::IoError,
    storage::{StorageHandle, StorageId, StorageUtilization},
};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::{Debug, Display};
use cubecl_environment::sync::Arc;

/// One device page, carved into [slices](Slice) of a fixed
/// [storage buffer](StorageHandle): the allocations on it and the free room
/// between them.
///
/// # Assumptions
///
/// `failures` is the [`ErrorGraph`] that holds the failures the page's slices
/// carry.
pub(crate) struct MemoryPage {
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

impl MemoryPage {
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

    /// Whether the page's storage id is backed by a real device allocation.
    pub(crate) fn is_mapped(&self) -> bool {
        self.mapped
    }

    /// Keep the page as it is for as long as the guard lives.
    pub(crate) fn guard(&self) -> PageGuard {
        PageGuard::page(&self.guards)
    }

    /// Whether a [guard](Self::guard) is keeping the page as it is.
    pub(crate) fn is_guarded(&self) -> bool {
        Arc::strong_count(&self.guards) > 1
    }

    /// The page's size in bytes.
    pub(crate) fn size(&self) -> u64 {
        self.storage.size()
    }

    /// Whether no allocation on the page is live.
    pub(crate) fn is_empty(&self) -> bool {
        self.slices.iter().all(Slice::is_free)
    }

    /// The slices holding a live allocation, in offset order.
    pub(crate) fn live(&self) -> impl Iterator<Item = &Slice> {
        live_slices(&self.slices)
    }

    /// The slice at `index`, mutably.
    ///
    /// # Panics
    ///
    /// When the page holds no slice at `index`.
    pub(crate) fn slice_at_index_mut(&mut self, index: usize) -> &mut Slice {
        &mut self.slices[index]
    }

    /// The storage behind the slice at `index`, while the page has real
    /// device backing: `None` for a page carved lazily and never resolved,
    /// which has nothing behind it to read.
    ///
    /// # Panics
    ///
    /// When the page holds no slice at `index`.
    pub(crate) fn mapped_storage_at_index(&self, index: usize) -> Option<StorageHandle> {
        mapped_storage(&self.slices, self.mapped, index)
    }

    /// The page-wide storage id, the one every slice on the page shares.
    pub(crate) fn storage_id(&self) -> StorageId {
        self.storage.id
    }

    /// Bind the page to the real device allocation `id`: the page and every
    /// slice take it (offsets and sizes are already correct — the virtual
    /// carving *is* the layout). The old minted id simply ceases to exist; it
    /// never reached the driver.
    pub(crate) fn rebind_storage(&mut self, id: StorageId) {
        rebind_slices(&mut self.slices, id);
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
        bind_at(&mut self.slices, &reserved, assigned, cursor, failures);
    }

    /// The page's usage: its live allocations and the bytes it holds.
    pub(crate) fn memory_usage(&self) -> MemoryUsage {
        slices_usage(self.slices.iter())
    }

    /// The page's free and full room, and each slice as a block when
    /// `with_blocks` asks for them.
    pub(crate) fn summary_with_blocks(&self, with_blocks: bool) -> MemoryPageSummary {
        summarize(&self.slices, self.size(), with_blocks)
    }

    /// Reserve `size_bytes` on the first free slice in offset order that
    /// fits them, padded, splitting off the room it leaves. `None` when no
    /// free slice fits them: [`coalesce`](Self::coalesce) merges neighbouring
    /// free slices into larger ones.
    pub(crate) fn try_reserve_bytes(&mut self, size_bytes: u64) -> Option<ManagedMemoryHandle> {
        carve(
            &mut self.slices,
            self.alignment,
            self.location_base,
            size_bytes,
        )
    }

    /// The slice `binding`'s location names.
    ///
    /// # Errors
    ///
    /// [`IoError::NotFound`] when the page holds no slice there.
    pub(crate) fn find(&self, binding: &ManagedMemoryBinding) -> Result<&Slice, IoError> {
        find_at(&self.slices, binding)
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
        find_at_mut(&mut self.slices, binding)
    }

    /// Number the page `page_index` in every location its slices carry.
    pub(crate) fn set_page_index(&mut self, page_index: u16) {
        self.location_base.page = page_index;
        number_slices(&self.slices, self.location_base);
    }

    /// Merge every run of neighbouring free slices into one, so larger
    /// allocations fit the page again. A merged slice is dropped here, so the
    /// failure it carried is released on the spot; nobody holds a handle to a
    /// free slice, so losing the tag costs nothing but the decrement.
    pub(crate) fn coalesce(&mut self, failures: &mut ErrorGraph) {
        coalesce(&mut self.slices, self.location_base, failures);
    }

    /// Release every failure the page's slices still carry, for a page about
    /// to be dropped whole, since a tag must not outlive its carrier. Answers
    /// the device allocation to give back, `None` when the page was never
    /// backed and there is nothing behind its minted id.
    pub(crate) fn release(&mut self, failures: &mut ErrorGraph) -> Option<StorageId> {
        shed(&mut self.slices, failures);
        self.mapped.then_some(self.storage.id)
    }
}

/// The slices of `slices` holding a live allocation.
fn live_slices(slices: &[Slice]) -> impl Iterator<Item = &Slice> {
    slices.iter().filter(|slice| !slice.is_free())
}

/// Point every slice of `slices` at the device allocation `id`.
fn rebind_slices(slices: &mut [Slice], id: StorageId) {
    for slice in slices {
        slice.storage.id = id;
    }
}

/// Bind `assigned` to the slice of `slices` that `reserved` names.
fn bind_at(
    slices: &mut [Slice],
    reserved: &ManagedMemoryHandle,
    assigned: ManagedMemoryHandle,
    cursor: u64,
    failures: &mut ErrorGraph,
) {
    let slice = slices
        .get_mut(reserved.descriptor().slice())
        .expect("a reserved slice should be on the page that reserved it");
    bind_slice(slice, assigned, cursor, failures);
}

/// The storage of the slice of `slices` at `index`, when the page is `mapped`.
fn mapped_storage(slices: &[Slice], mapped: bool, index: usize) -> Option<StorageHandle> {
    mapped.then(|| slices[index].storage.clone())
}

/// The slice of `slices` `binding`'s location names.
fn find_at<'a>(slices: &'a [Slice], binding: &ManagedMemoryBinding) -> Result<&'a Slice, IoError> {
    let index = binding.descriptor().slice();
    slices.get(index).ok_or_else(|| missing("slice", index))
}

/// The slice of `slices` `binding`'s location names, mutably.
fn find_at_mut<'a>(
    slices: &'a mut [Slice],
    binding: &ManagedMemoryBinding,
) -> Result<&'a mut Slice, IoError> {
    let index = binding.descriptor().slice();
    slices.get_mut(index).ok_or_else(|| missing("slice", index))
}

/// Merge every run of neighbouring free slices of `slices` into one, and
/// number them from `location_base` again.
fn coalesce(slices: &mut Vec<Slice>, location_base: MemoryLocation, failures: &mut ErrorGraph) {
    slices.dedup_by(|later, earlier| merge_free(later, earlier, failures));
    number_slices(slices, location_base);
}

/// Give every slice of `slices` the location `location_base` with its index.
fn number_slices(slices: &[Slice], location_base: MemoryLocation) {
    for (index, slice) in slices.iter().enumerate() {
        slice.descriptor().update_location(MemoryLocation {
            slice: index as u32,
            ..location_base
        });
    }
}

/// Release every failure the slices of `slices` carry.
fn shed(slices: &mut [Slice], failures: &mut ErrorGraph) {
    for slice in slices {
        slice.tainted.clear(failures);
    }
}

/// Merge `later` into `earlier` when both are free, answering whether it did.
/// The merged slice is a fresh one over both, and both release the failures
/// they carried.
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

/// Reserve `size` bytes, padded to `alignment`, on the first free slice of
/// `slices` that fits them, splitting off the room it leaves after it.
fn carve(
    slices: &mut Vec<Slice>,
    alignment: u64,
    location_base: MemoryLocation,
    size: u64,
) -> Option<ManagedMemoryHandle> {
    let effective_size = padded_size(size, alignment)?;
    let index = slices
        .iter()
        .position(|slice| slice.is_free() && slice.effective_size() >= effective_size)?;
    let slice = &mut slices[index];
    let rest = split(slice, size, effective_size);
    let handle = slice.handle.clone();
    if let Some(rest) = rest {
        insert_after(slices, index, rest, location_base);
    }
    Some(handle)
}

/// Serve `size` bytes, `effective_size` once padded, from the start of the
/// free `slice`, whose room is its padded size: the padding its last
/// allocation took is part of it. Answers the slice of the room left after
/// them, if any is.
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

/// Place `slice` right after the slice at `index`. Only the slices after it
/// move, so only their descriptors learn a new position: a reservation splits
/// a slice on every launch that allocates, and a page holds many.
fn insert_after(
    slices: &mut Vec<Slice>,
    index: usize,
    slice: Slice,
    location_base: MemoryLocation,
) {
    slices.insert(index + 1, slice);
    for (position, slice) in slices.iter().enumerate().skip(index + 1) {
        slice.descriptor().update_location(MemoryLocation {
            slice: position as u32,
            ..location_base
        });
    }
}

/// The summary of a page of `total_bytes` carved into `slices`.
fn summarize(slices: &[Slice], total_bytes: u64, with_blocks: bool) -> MemoryPageSummary {
    let free = || slices.iter().filter(|slice| slice.is_free());
    let full = || live_slices(slices);
    MemoryPageSummary {
        blocks: slices
            .iter()
            .filter(|_| with_blocks)
            .map(|slice| MemoryBlock {
                is_free: slice.is_free(),
                size_bytes: slice.effective_size(),
            })
            .collect(),
        free_bytes: free().map(Slice::effective_size).sum(),
        full_bytes: full().map(Slice::effective_size).sum(),
        total_bytes,
        free_count: free().count(),
        full_count: full().count(),
        total_count: slices.len(),
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
    /// The page's slices in offset order (Position), when asked for.
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
        let state = match self.is_free {
            true => "Free",
            false => "Reserved",
        };
        write!(
            f,
            "{state} ({})",
            BytesFormat::from_byte_count(self.size_bytes)
        )
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

impl Display for MemoryPage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{}", self.summary_with_blocks(true)))
    }
}

#[cfg(test)]
#[allow(clippy::identity_op)]
mod tests {
    use super::*;
    use alloc::vec;

    const MB: u64 = 1024 * 1024;

    fn block(is_free: bool, size_bytes: u64) -> MemoryBlock {
        MemoryBlock {
            is_free,
            size_bytes,
        }
    }

    fn reserve(page: &mut MemoryPage, size_bytes: u64) -> ManagedMemoryHandle {
        page.try_reserve_bytes(size_bytes)
            .expect("the page should have room for the slice")
    }

    fn offset_of(page: &MemoryPage, handle: &ManagedMemoryHandle) -> StorageUtilization {
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
        assert_eq!(page.summary_with_blocks(true).total_count, 2);
    }

    /// Coalescing merges the free slices a freed reservation leaves back into
    /// one.
    #[test_log::test]
    fn coalescing_merges_neighbouring_free_slices() {
        let mut page = new_memory_page(32 * MB);
        core::mem::drop(reserve(&mut page, 16 * MB));

        page.coalesce(&mut ErrorGraph::default());

        assert_eq!(
            page.summary_with_blocks(true),
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

    /// Reservations fill a page in offset order, and one that fits no free
    /// slice is refused.
    #[test_log::test]
    fn reservations_fill_a_page_in_order() {
        let mut page = new_memory_page(32 * MB);
        let _slices = [4, 15, 8, 4].map(|size| reserve(&mut page, size * MB));

        assert_eq!(
            page.summary_with_blocks(true),
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
        assert!(page.try_reserve_bytes(8 * MB).is_none());
    }

    /// A freed slice serves only what fits it until coalescing merges it with
    /// its free neighbour, and the live slices keep their offsets.
    #[test_log::test]
    fn a_freed_slice_serves_more_once_coalesced() {
        let mut page = new_memory_page(32 * MB);
        let [_first, second, third, fourth] =
            [4, 15, 8, 4].map(|size| reserve(&mut page, size * MB));
        core::mem::drop(second);
        let _nine = reserve(&mut page, 9 * MB);
        core::mem::drop(third);
        assert!(page.try_reserve_bytes(9 * MB).is_none(), "6 and 8 MB apart");

        page.coalesce(&mut ErrorGraph::default());

        assert!(page.try_reserve_bytes(9 * MB).is_some(), "14 MB together");
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
            page.try_reserve_bytes(100).is_some(),
            "the slice is 128 bytes"
        );
    }

    /// Splitting a freed slice leaves the rest of its padded room on the page.
    #[test_log::test]
    fn splitting_a_freed_slice_keeps_its_padding_on_the_page() {
        let mut page = aligned_memory_page(128, 32);
        core::mem::drop(reserve(&mut page, 100));

        let _split = reserve(&mut page, 32);

        assert_eq!(page.memory_usage().reserved_bytes, 128);
    }

    /// A size whose padding passes the largest `u64` fits no slice.
    #[test_log::test]
    fn a_size_whose_padding_overflows_fits_no_slice() {
        let mut page = aligned_memory_page(128, 32);

        assert!(page.try_reserve_bytes(u64::MAX).is_none());
    }

    /// A binding takes the slice's location and the cursor it is bound at.
    #[test_log::test]
    fn a_binding_takes_the_slice_and_its_cursor() {
        let mut page = new_memory_page(32 * MB);
        let reserved = reserve(&mut page, MB);
        let assigned = ManagedMemoryHandle::new();

        page.bind(reserved, assigned.clone(), 3, &mut ErrorGraph::default());

        let slice = page
            .find(&assigned.clone().binding())
            .expect("the binding should resolve");
        assert_eq!(slice.cursor, 3);
    }

    /// A lazily carved page reads no storage until it is rebound to a real
    /// allocation.
    #[test_log::test]
    fn a_lazy_page_has_storage_only_once_rebound() {
        let storage = StorageHandle::new(
            StorageId::new(),
            StorageUtilization {
                offset_bytes: 0,
                size_bytes: MB,
            },
        );
        let mut page = MemoryPage::new(storage, 4, MemoryLocation::new(0, 0, 0), PageMapping::Lazy);
        assert!(page.mapped_storage_at_index(0).is_none());

        let real = StorageId::new();
        page.rebind_storage(real);

        let mapped = page
            .mapped_storage_at_index(0)
            .expect("a rebound page should have storage");
        assert_eq!(mapped.id, real);
    }

    fn new_memory_page(size: u64) -> MemoryPage {
        aligned_memory_page(size, 4)
    }

    fn aligned_memory_page(size: u64, alignment: u64) -> MemoryPage {
        let storage = StorageHandle::new(
            StorageId::new(),
            StorageUtilization {
                offset_bytes: 0,
                size_bytes: size,
            },
        );

        MemoryPage::new(
            storage,
            alignment,
            MemoryLocation::new(0, 0, 0),
            PageMapping::Eager,
        )
    }
}
