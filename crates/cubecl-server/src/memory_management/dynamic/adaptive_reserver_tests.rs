//! The Medium Tests of the adaptive memory: they run one at a time with the
//! tests that open a process mode override, since one that discards launches
//! maps pages lazily, so the pages these tests reserve would hold no device
//! bytes and the memory pressure they measure would read as none.
//!
//! The layout these tests drive is the `Adaptive` preset, which a build
//! refusing sub-slicing does not have.

// `serial_test`'s macro expands to `vec!`, which a `no_std` crate has to
// bring in itself.
use crate::{
    logging::ServerLogger,
    memory_management::{
        AllocationKind, ErrorGraph, ManagedMemoryHandle, MemoryAllocationMode, MemoryConfiguration,
        MemoryManagement, MemoryManagementOptions, MemoryPoolKind, PageUpdate,
        relocation::{
            HostRelocationCopier, RelocationCopier, RelocationNeed, RelocationReason, StorageCopy,
        },
    },
    server::{IoError, ServerError},
    storage::BytesStorage,
};
use alloc::vec;
use cubecl_environment::sync::Arc;
use cubecl_ir::MemoryDeviceProperties;

const KIB: u64 = 1024;
const MIB: u64 = 1024 * KIB;
const PROPERTIES: MemoryDeviceProperties = MemoryDeviceProperties::new(1024 * MIB, 32);

/// The floor the `Adaptive` preset carves its pages at, until a larger
/// allocation grows them.
const FLOOR_BYTES: u64 = 2 * MIB;

/// A memory management laid out by the `Adaptive` preset, on a device
/// that holds `max_memory` bytes.
fn adaptive_on_device(max_memory: u64) -> MemoryManagement<BytesStorage> {
    MemoryManagement::from_configuration(
        BytesStorage::default(),
        &PROPERTIES.with_max_memory(max_memory),
        MemoryConfiguration::Adaptive,
        Arc::new(ServerLogger::default()),
        MemoryManagementOptions::new("adaptive"),
    )
}

/// A memory management laid out by the `Adaptive` preset. Every size these
/// tests reserve is past the metadata pool's largest slice, so the arena
/// is what serves them.
fn adaptive() -> MemoryManagement<BytesStorage> {
    MemoryManagement::from_configuration(
        BytesStorage::default(),
        &PROPERTIES,
        MemoryConfiguration::Adaptive,
        Arc::new(ServerLogger::default()),
        MemoryManagementOptions::new("adaptive"),
    )
}

/// The adaptive pool, as its report states it.
#[derive(Debug, PartialEq, Eq)]
struct Pool {
    page_size: u64,
    pages: u64,
    outdated: u64,
}

/// The adaptive pool: the size it carves pages at, every page it holds —
/// outdated ones included — and how many of those are outdated.
fn pool(memory: &MemoryManagement<BytesStorage>) -> Pool {
    let report = memory.memory_report().dynamic;
    let current = report
        .iter()
        .find_map(|pool| match pool.kind {
            MemoryPoolKind::Adaptive {
                page_size,
                outdated_pages,
            } => Some((page_size, outdated_pages)),
            _ => None,
        })
        .expect("the layer should carve allocations from adaptive pages");
    Pool {
        page_size: current.0,
        pages: report.iter().map(|pool| pool.page_count).sum(),
        outdated: current.1,
    }
}

fn reserve(memory: &mut MemoryManagement<BytesStorage>, size: u64) -> ManagedMemoryHandle {
    memory
        .reserve(size, PageUpdate::Allow, &mut ErrorGraph::default())
        .expect("the memory should serve the reservation")
}

/// Where an allocation sits: its pool, page and slice.
fn place(handle: &ManagedMemoryHandle) -> (u8, u16, u32) {
    let location = handle.descriptor().location();
    (location.pool, location.page, location.slice)
}

/// The page an allocation sits on, which only its pool numbers.
fn page_of(handle: &ManagedMemoryHandle) -> (u8, u16) {
    let location = handle.descriptor().location();
    (location.pool, location.page)
}

fn fill(memory: &mut MemoryManagement<BytesStorage>, handle: &ManagedMemoryHandle, byte: u8) {
    let mut resource = memory
        .get_resource(handle.clone().binding(), None, None)
        .expect("the binding should resolve");
    resource.write().fill(byte);
}

fn contents(memory: &mut MemoryManagement<BytesStorage>, handle: &ManagedMemoryHandle) -> u8 {
    let resource = memory
        .get_resource(handle.clone().binding(), None, None)
        .expect("the binding should resolve");
    let bytes = resource.read();
    assert!(bytes.iter().all(|byte| *byte == bytes[0]));
    bytes[0]
}

/// Run a relocation the way a command does, with a host copy standing in
/// for the device one. Answers how many allocations moved.
fn relocate(memory: &mut MemoryManagement<BytesStorage>) -> usize {
    let failures = &mut ErrorGraph::default();
    let before = memory.memory_report().dynamic.len();
    let mut copies = HostRelocationCopier;
    memory.relocate(&mut copies, RelocationReason::Explicit, failures);
    // A pool a relocation emptied is dropped, so the reports it leaves say
    // how many moved off it.
    before - memory.memory_report().dynamic.len()
}

/// A device that refuses every copy, which abandons the relocation.
struct RefusedCopies;

// SAFETY: test memory that no work reads or writes, and copies that do
// nothing or land on return.
unsafe impl RelocationCopier<BytesStorage> for RefusedCopies {
    fn wait_device(&mut self) -> Result<(), ServerError> {
        Ok(())
    }

    unsafe fn copy(
        &mut self,
        _storage: &mut BytesStorage,
        _copy: &StorageCopy,
    ) -> Result<(), IoError> {
        Err(IoError::Unknown {
            description: "refused".into(),
            backtrace: Default::default(),
        })
    }

    fn wait_copies(&mut self) -> Result<(), ServerError> {
        Ok(())
    }
}

/// Whether a relocation is wanted, as the server asks before a
/// reservation, with this memory the only one on the device.
fn relocation(memory: &MemoryManagement<BytesStorage>) -> Option<RelocationReason> {
    match memory
        .relocation_need()
        .resolve(|| memory.bytes_allocated())
    {
        RelocationNeed::Wanted(reason) => Some(reason),
        _ => None,
    }
}

/// A device with room to spare says nothing; one whose next page would
/// leave it less than another page asks for the outdated pools to be
/// emptied first, which is the last moment a relocation has somewhere to
/// copy to.
#[test]
#[serial_test::serial(records)]
fn memory_pressure_asks_for_a_relocation_before_the_next_page() {
    const MAX_MEMORY: u64 = 24 * MIB;
    let mut memory = adaptive_on_device(MAX_MEMORY);

    let kept = reserve(&mut memory, MIB);
    assert_eq!(relocation(&memory), None, "one 2 MiB page of 24 MiB");

    // 13 MiB held over two pools, and the next page is 11 MiB.
    let large = reserve(&mut memory, 10 * MIB);
    assert_eq!(relocation(&memory), Some(RelocationReason::MemoryPressure));

    let mut copies = HostRelocationCopier;
    memory.relocate(
        &mut copies,
        RelocationReason::MemoryPressure,
        &mut ErrorGraph::default(),
    );
    assert_eq!(
        pool(&memory),
        Pool {
            page_size: 11 * MIB,
            pages: 1,
            outdated: 0
        },
        "the outdated pool was emptied into the room the current page has"
    );
    assert_eq!(page_of(&kept), page_of(&large));
    assert_eq!(relocation(&memory), None, "nothing is outdated any more");
}

/// The bytes the pools hold in pages, every pool together.
fn reserved(memory: &MemoryManagement<BytesStorage>) -> u64 {
    memory
        .memory_report()
        .dynamic
        .iter()
        .map(|pool| pool.usage.reserved_bytes)
        .sum()
}

/// A buffer kept outside a capture, which the next graph reads — a recording's input — takes
/// no arena page of its own once the stream's graphs pin every page the arena holds. Each such
/// page would hold that one buffer, and the graph reading it would pin it in turn, so every
/// recording would add a page of the arena's size, however small its input.
#[test]
#[serial_test::serial(records)]
fn a_kept_input_takes_no_arena_page_once_graphs_pin_the_arena() {
    let mut memory = adaptive();
    // A graph's temporary sizes the arena's pages, and the graph pins its page.
    let temporary = reserve(&mut memory, 8 * MIB);
    let _graph = memory.guard(temporary.descriptor().location());
    drop(temporary);
    let before = reserved(&memory);

    // Each later graph's input, read by that graph and kept for its life.
    let inputs: alloc::vec::Vec<_> = (0..4)
        .map(|_| {
            let input = reserve(&mut memory, 128 * 1024);
            let graph = memory.guard(input.descriptor().location());
            (input, graph)
        })
        .collect();

    let grown = reserved(&memory) - before;
    assert!(
        grown < 9 * MIB,
        "four 128 KiB inputs took {grown} B, an arena page each"
    );
    drop(inputs);
}

/// Metadata a graph keeps and small inputs it reads, kept for its life, pack together rather
/// than taking a metadata page each: once graphs pin every page of the small-allocation pool,
/// each would otherwise take a page of its own, which the graph reading it pins in turn and
/// which never empties, a page per graph.
#[test]
#[serial_test::serial(records)]
fn kept_metadata_and_small_inputs_take_no_small_page_per_graph() {
    let mut memory = adaptive();
    let failures = &mut ErrorGraph::default();
    // A graph's temporary metadata, on the small pool's one page, which the graph pins.
    let temporary = reserve_kind(
        &mut memory,
        KIB,
        PageUpdate::Allow,
        AllocationKind::GraphIntermediate,
    )
    .expect("the memory should serve the reservation");
    let _graph = memory.guard(temporary.descriptor().location());
    drop(temporary);
    let mut kept = alloc::vec::Vec::new();
    let before = reserved(&memory);

    for _ in 0..4 {
        // What a recording keeps for its graph: its launches' metadata and its input.
        let metadata = memory
            .reserve_as(
                KIB,
                PageUpdate::Allow,
                AllocationKind::GraphOutput,
                failures,
            )
            .expect("the memory should serve the reservation");
        let input = reserve(&mut memory, 4 * KIB);
        for handle in [metadata, input] {
            let graph = memory.guard(handle.descriptor().location());
            kept.push((handle, graph));
        }
    }

    let grown = reserved(&memory) - before;
    assert!(
        grown <= 8 * MIB,
        "four graphs' metadata and small inputs took {grown} B, a metadata page each"
    );
}

/// Reserve `size` bytes of `kind`, changing the pages held as far as `update` allows.
fn reserve_kind(
    memory: &mut MemoryManagement<BytesStorage>,
    size: u64,
    update: PageUpdate,
    kind: AllocationKind,
) -> Result<ManagedMemoryHandle, IoError> {
    memory.reserve_as(size, update, kind, &mut ErrorGraph::default())
}

/// Work outside a capture takes the room graphs' outputs keep apart only once graphs pin the
/// arena: while the arena has a page nothing pins, a reservation that may not add one finds
/// no room, where one that may adds an arena page.
#[test]
#[serial_test::serial(records)]
fn eager_work_takes_no_kept_apart_room_while_the_arena_is_unpinned() {
    let mut memory = adaptive();
    // A 9 MiB arena page nothing pins, with 1 MiB left, and an output's 2 MiB page apart.
    let _held = reserve(&mut memory, 8 * MIB);
    let _output = reserve_kind(
        &mut memory,
        128 * KIB,
        PageUpdate::Allow,
        AllocationKind::GraphOutput,
    );

    let placed = reserve_kind(
        &mut memory,
        3 * MIB / 2,
        PageUpdate::Forbidden,
        AllocationKind::Eager,
    );

    assert!(
        matches!(placed, Err(IoError::PageUpdateForbidden { .. })),
        "placed on {:?}",
        placed.as_ref().map(place)
    );
}

/// Small work outside a capture takes the room kept apart for graphs' metadata only once
/// graphs pin every page of the small pool: a full page nothing pins gets a new small page.
#[test]
#[serial_test::serial(records)]
fn small_eager_work_takes_no_kept_apart_room_while_the_small_pool_is_unpinned() {
    let mut memory = adaptive();
    let _full: alloc::vec::Vec<_> = (0..128).map(|_| reserve(&mut memory, 64 * KIB)).collect();
    let metadata = reserve_kind(
        &mut memory,
        KIB,
        PageUpdate::Allow,
        AllocationKind::GraphOutput,
    )
    .expect("the memory should serve the reservation");

    let input = reserve(&mut memory, 4 * KIB);

    assert_ne!(
        input.descriptor().location().pool,
        metadata.descriptor().location().pool
    );
}

/// Work outside a capture never grows the separate arena's pages, whose slots nothing frees:
/// once graphs pin the arena, a buffer the separate arena's pages do not fit takes an arena
/// page, as it would before.
#[test]
#[serial_test::serial(records)]
fn eager_work_never_grows_the_separate_arena() {
    let mut memory = adaptive();
    // A graph's temporary sizes the arena's pages at 9 MiB, and the graph pins its page. The
    // separate arena carves at its 2 MiB floor.
    let temporary = reserve(&mut memory, 8 * MIB);
    let _graph = memory.guard(temporary.descriptor().location());
    drop(temporary);

    let input = reserve(&mut memory, 3 * MIB);

    assert!(
        (super::adaptive_reserver::ARENA_POOLS..super::adaptive_reserver::SEPARATE_POOLS)
            .contains(&input.descriptor().location().pool),
        "placed on {:?}",
        place(&input)
    );
}

/// A small buffer a planning run carved on the room a graph pinned, and that the work kept,
/// moves to the small pool kept apart with its bytes, as an arena buffer moves to the
/// separate arena: left there, the graph's next replay would write over it.
#[test]
#[serial_test::serial(records)]
fn a_kept_small_buffer_on_a_pinned_page_moves_apart() {
    let mut memory = adaptive();
    let temporary = reserve_kind(
        &mut memory,
        KIB,
        PageUpdate::Allow,
        AllocationKind::GraphIntermediate,
    )
    .expect("the memory should serve the reservation");
    let _graph = memory.guard(temporary.descriptor().location());
    drop(temporary);
    let kept = reserve_kind(
        &mut memory,
        4 * KIB,
        PageUpdate::Allow,
        AllocationKind::GraphIntermediate,
    )
    .expect("the memory should serve the reservation");
    fill(&mut memory, &kept, 7);

    memory
        .move_apart(
            &[kept.clone().binding()],
            &mut HostRelocationCopier,
            &mut ErrorGraph::default(),
        )
        .expect("the kept buffer should move apart");

    assert_eq!(
        (
            kept.descriptor().location().pool,
            contents(&mut memory, &kept)
        ),
        (super::adaptive_reserver::SMALL_SEPARATE_POOL, 7)
    );
}

/// With nothing outdated there is nothing to free, however full the
/// device: no relocation, and no device wait to pay for one.
#[test]
#[serial_test::serial(records)]
fn a_full_device_with_nothing_outdated_asks_for_nothing() {
    let mut memory = adaptive_on_device(4 * MIB);
    let _held = reserve(&mut memory, MIB);
    assert_eq!(relocation(&memory), None);
}

/// A relocation that found nothing to move is not planned again until
/// the arena changes: the next reservations under the same pressure pay
/// for no plan.
#[test]
#[serial_test::serial(records)]
fn a_relocation_that_moved_nothing_is_not_planned_again() {
    let mut memory = adaptive_on_device(24 * MIB);

    let recorded = reserve(&mut memory, MIB);
    let guard = memory.guard(recorded.descriptor().location());
    let _large = reserve(&mut memory, 10 * MIB);
    assert_eq!(relocation(&memory), Some(RelocationReason::MemoryPressure));

    memory.relocate(
        &mut HostRelocationCopier,
        RelocationReason::MemoryPressure,
        &mut ErrorGraph::default(),
    );
    assert_eq!(relocation(&memory), None, "the guarded page could not move");

    drop(guard);
    assert_eq!(
        relocation(&memory),
        Some(RelocationReason::MemoryPressure),
        "a released guard is a page the last plan could not move"
    );
}

/// A new page is room the last plan did not have, so it plans again.
#[test]
#[serial_test::serial(records)]
fn a_new_page_lets_a_stalled_relocation_plan_again() {
    let mut memory = adaptive_on_device(24 * MIB);

    let recorded = reserve(&mut memory, MIB);
    let _guard = memory.guard(recorded.descriptor().location());
    let _large = reserve(&mut memory, 10 * MIB);
    memory.relocate(
        &mut HostRelocationCopier,
        RelocationReason::MemoryPressure,
        &mut ErrorGraph::default(),
    );
    assert_eq!(relocation(&memory), None);

    let _more = reserve(&mut memory, 10 * MIB);
    assert_eq!(relocation(&memory), Some(RelocationReason::MemoryPressure));
}

/// An add-only reservation releases nothing, even an outdated page that
/// emptied: while a graph records, the pages it touched must keep their
/// numbers and addresses until it seals.
#[test]
#[serial_test::serial(records)]
fn an_add_only_reservation_releases_nothing() {
    let mut memory = adaptive();

    let first = reserve(&mut memory, MIB);
    let _large = reserve(&mut memory, 10 * MIB);
    drop(first);

    let _kept = memory
        .reserve(MIB, PageUpdate::AddOnly, &mut ErrorGraph::default())
        .expect("the memory should serve the reservation");
    assert_eq!(pool(&memory).outdated, 1, "the emptied page is still held");

    let _tick = reserve(&mut memory, MIB);
    assert_eq!(pool(&memory).outdated, 0, "a plain reservation releases it");
}

/// The page size follows the largest allocation the pool served: a
/// megabyte of slack, rounded to the megabyte, never below the floor.
#[test]
#[serial_test::serial(records)]
fn the_page_size_follows_the_largest_allocation() {
    let mut memory = adaptive();

    let _small = reserve(&mut memory, MIB);
    assert_eq!(
        pool(&memory).page_size,
        FLOOR_BYTES,
        "the floor holds small workloads"
    );

    let _large = reserve(&mut memory, 10 * MIB);
    assert_eq!(pool(&memory).page_size, 11 * MIB);

    let _smaller = reserve(&mut memory, 6 * MIB);
    assert_eq!(
        pool(&memory).page_size,
        11 * MIB,
        "the page size never shrinks"
    );
}

/// The page size stops at the largest page the device allocates, and an
/// allocation no such page fits is no pool's to serve.
#[test]
#[serial_test::serial(records)]
fn the_page_size_stops_at_the_devices_largest_page() {
    const MAX_PAGE: u64 = 16 * MIB;
    let mut memory = MemoryManagement::from_configuration(
        BytesStorage::default(),
        &MemoryDeviceProperties::new(MAX_PAGE, 32),
        MemoryConfiguration::Adaptive,
        Arc::new(ServerLogger::default()),
        MemoryManagementOptions::new("adaptive"),
    );

    let _largest = reserve(&mut memory, MAX_PAGE);
    assert_eq!(
        pool(&memory).page_size,
        MAX_PAGE,
        "the slack does not push the page past the device's limit"
    );

    let refused = memory.reserve(MAX_PAGE + 1, PageUpdate::Allow, &mut ErrorGraph::default());
    assert!(
        matches!(refused, Err(IoError::BufferTooBig { .. })),
        "{refused:?}"
    );
}

/// An allocation no page fits is refused before anything changes: the
/// pages held keep serving. (A refusal by the device itself is the
/// arena's to test, with a storage that refuses.)
#[test]
#[serial_test::serial(records)]
fn an_allocation_no_page_fits_leaves_the_pages_alone() {
    let mut memory = adaptive();
    let _small = reserve(&mut memory, MIB);

    let refused = memory.reserve(1 << 50, PageUpdate::Allow, &mut ErrorGraph::default());
    assert!(
        matches!(refused, Err(IoError::BufferTooBig { .. })),
        "{refused:?}"
    );
    assert_eq!(
        pool(&memory),
        Pool {
            page_size: FLOOR_BYTES,
            pages: 1,
            outdated: 0
        }
    );

    let _served = reserve(&mut memory, MIB);
    assert_eq!(pool(&memory).pages, 1, "served from the page already held");
}

/// Growing outdates every page held: none of them serves another
/// reservation, even with room to spare.
#[test]
#[serial_test::serial(records)]
fn a_grown_page_size_serves_nothing_from_outdated_pages() {
    let mut memory = adaptive();

    let first = reserve(&mut memory, MIB);
    let _large = reserve(&mut memory, 10 * MIB);
    assert_eq!(
        pool(&memory),
        Pool {
            page_size: 11 * MIB,
            pages: 2,
            outdated: 1
        }
    );

    let small = reserve(&mut memory, MIB);
    assert_ne!(
        page_of(&small),
        page_of(&first),
        "the outdated page has room to spare, but it is outdated"
    );
}

/// An outdated page goes back to the driver on the tick after its last
/// slice is freed.
#[test]
#[serial_test::serial(records)]
fn an_outdated_page_is_released_once_empty() {
    let mut memory = adaptive();

    let first = reserve(&mut memory, MIB);
    let _large = reserve(&mut memory, 10 * MIB);
    assert_eq!(
        pool(&memory),
        Pool {
            page_size: 11 * MIB,
            pages: 2,
            outdated: 1
        }
    );

    drop(first);
    let _tick = reserve(&mut memory, MIB);
    assert_eq!(
        pool(&memory),
        Pool {
            page_size: 11 * MIB,
            pages: 1,
            outdated: 0
        }
    );
}

/// A reservation that may not update the pages serves only from the room
/// they hold, and says so when that room is not enough.
#[test]
#[serial_test::serial(records)]
fn a_forbidden_page_update_serves_only_the_room_held() {
    let mut memory = adaptive();
    let first = reserve(&mut memory, MIB);
    drop(first);

    let _held = memory
        .reserve(MIB, PageUpdate::Forbidden, &mut ErrorGraph::default())
        .expect("the page held should have room for it");
    let refused = memory.reserve(4 * MIB, PageUpdate::Forbidden, &mut ErrorGraph::default());
    assert!(
        matches!(refused, Err(IoError::PageUpdateForbidden { .. })),
        "{refused:?}"
    );
    assert_eq!(pool(&memory).pages, 1, "no page was added");
}

/// Relocation moves live allocations off outdated pages with their bytes,
/// their owners resolve to the new place, and the outdated pages go back.
#[test]
#[serial_test::serial(records)]
fn relocation_moves_live_allocations_to_current_pages() {
    let mut memory = adaptive();

    let kept = reserve(&mut memory, MIB);
    fill(&mut memory, &kept, 7);
    let large = reserve(&mut memory, 10 * MIB);
    assert_eq!(
        pool(&memory),
        Pool {
            page_size: 11 * MIB,
            pages: 2,
            outdated: 1
        }
    );

    assert_eq!(relocate(&mut memory), 1);

    assert_eq!(
        pool(&memory),
        Pool {
            page_size: 11 * MIB,
            pages: 1,
            outdated: 0
        },
        "the outdated page is gone"
    );
    assert_eq!(contents(&mut memory, &kept), 7, "the bytes moved with it");
    assert_eq!(
        page_of(&kept),
        page_of(&large),
        "it shares the current page"
    );
    assert_eq!(memory.memory_report().usage().in_use_bytes, 11 * MIB);
}

/// A plan dropped before its commit leaves every allocation where it was.
#[test]
#[serial_test::serial(records)]
fn an_abandoned_relocation_loses_nothing() {
    let mut memory = adaptive();

    let kept = reserve(&mut memory, MIB);
    fill(&mut memory, &kept, 3);
    let location = place(&kept);
    let _large = reserve(&mut memory, 10 * MIB);

    let mut refused = RefusedCopies;
    memory.relocate(
        &mut refused,
        RelocationReason::Explicit,
        &mut ErrorGraph::default(),
    );

    assert_eq!(place(&kept), location);
    assert_eq!(contents(&mut memory, &kept), 3);
}

/// A guarded page keeps its address: a recorded graph replays against it.
#[test]
#[serial_test::serial(records)]
fn relocation_leaves_guarded_pages_in_place() {
    let mut memory = adaptive();

    let recorded = reserve(&mut memory, MIB);
    let location = place(&recorded);
    let _guard = memory.guard(recorded.descriptor().location());

    let _large = reserve(&mut memory, 10 * MIB);
    assert_eq!(relocate(&mut memory), 0);
    assert_eq!(place(&recorded), location);
}

/// A guard holds the whole page: everything on it stays.
#[test]
#[serial_test::serial(records)]
fn a_guard_holds_its_whole_page() {
    let mut memory = adaptive();

    let recorded = reserve(&mut memory, MIB);
    let neighbour = reserve(&mut memory, MIB / 2);
    let location = place(&neighbour);
    assert_eq!(page_of(&neighbour), page_of(&recorded));
    let _guard = memory.guard(recorded.descriptor().location());

    let _large = reserve(&mut memory, 10 * MIB);
    assert_eq!(relocate(&mut memory), 0);
    assert_eq!(place(&neighbour), location);
}

/// A guarded page hands out nothing new, even where its memory is free.
#[test]
#[serial_test::serial(records)]
fn a_guarded_page_serves_no_new_reservation() {
    let mut memory = adaptive();

    let recorded = reserve(&mut memory, MIB);
    let guard = memory.guard(recorded.descriptor().location());
    drop(recorded);

    let later = reserve(&mut memory, MIB);
    assert_eq!(
        pool(&memory).pages,
        2,
        "the guarded page stays out of reach"
    );
    drop(guard);
    drop(later);
}

/// Only what the pool serves counts: persistent and dedicated allocations,
/// however large, leave its page size alone.
#[test]
#[serial_test::serial(records)]
fn other_pools_do_not_move_the_statistic() {
    let mut memory = adaptive();
    let _dynamic = reserve(&mut memory, MIB);

    memory.mode(MemoryAllocationMode::Persistent);
    let _weight = reserve(&mut memory, 100 * MIB);
    memory.mode(MemoryAllocationMode::Auto);

    memory.mode(MemoryAllocationMode::Dedicated);
    let _probe = reserve(&mut memory, 200 * MIB);
    memory.mode(MemoryAllocationMode::Auto);

    assert_eq!(
        pool(&memory),
        Pool {
            page_size: FLOOR_BYTES,
            pages: 1,
            outdated: 0
        }
    );
}

/// A guard's claim ends with the guard: the page can move again.
#[test]
#[serial_test::serial(records)]
fn a_dropped_guard_lets_the_page_move() {
    let mut memory = adaptive();

    let recorded = reserve(&mut memory, MIB);
    drop(memory.guard(recorded.descriptor().location()));

    let _large = reserve(&mut memory, 10 * MIB);
    assert_eq!(relocate(&mut memory), 1, "the page moves once unguarded");
    drop(recorded);
}
