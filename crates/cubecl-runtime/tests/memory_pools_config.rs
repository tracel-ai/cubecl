//! End-to-end test of the programmatic pool layout: the whole path from a
//! [`MemoryPoolsConfig`] payload through [`MemoryConfiguration::resolve`] to
//! the actual pool behavior, plus the in-place rebuild
//! ([`MemoryManagement::install_pools`]) that re-sizes the pools between
//! workloads. There is deliberately no config-file pathway for pool layouts —
//! they are dynamic, set at runtime per workload.

// Every test here exercises sliced pools, which `exclusive_memory_only`
// builds reject by design.
#![cfg(not(exclusive_memory_only))]

use std::sync::Arc;

use cubecl_ir::MemoryDeviceProperties;
use cubecl_runtime::config::memory::{MemoryPoolConfig, MemoryPoolsConfig};
use cubecl_runtime::config::size::MemorySize;
use cubecl_runtime::dry_run::{DryRun, RealRun};
use cubecl_runtime::logging::ServerLogger;
use cubecl_runtime::memory_management::{
    InstallMemoryPoolsError, MemoryAllocationMode, MemoryConfiguration, MemoryManagement,
    MemoryManagementOptions, MemoryPoolKind,
};
use cubecl_runtime::storage::BytesStorage;

const MIB: u64 = 1024 * 1024;

fn sliced(page_size: u64, pages: u64) -> MemoryPoolsConfig {
    MemoryPoolsConfig::Explicit(vec![MemoryPoolConfig::Sliced {
        page_size: MemorySize(page_size),
        max_slice_size: None,
        max_pool_size: Some(MemorySize(page_size * pages)),
        dealloc_period: None,
    }])
}

fn props() -> MemoryDeviceProperties {
    MemoryDeviceProperties {
        max_page_size: 128 * MIB,
        alignment: 32,
    }
}

#[test]
fn programmatic_pools_override_runtime_default() {
    // The path a downstream user takes when the budget is computed at runtime
    // (e.g. an LLM activation working set), and what every runtime does for
    // its main GPU pool at stream creation.
    let pools = sliced(MIB, 2);
    let resolved = MemoryConfiguration::default()
        .resolve(Some(&pools), &props())
        .unwrap();
    let mut memory_management = MemoryManagement::from_configuration(
        BytesStorage::default(),
        &props(),
        resolved,
        Arc::new(ServerLogger::default()),
        MemoryManagementOptions::new("Main GPU Memory"),
    );

    // Vastly different sizes share the same arena instead of landing in
    // size-bucketed pools with separate reservations.
    let small = memory_management.reserve(4096).unwrap();
    drop(small);
    let _large = memory_management.reserve(512 * 1024).unwrap();
    assert_eq!(memory_management.memory_usage().bytes_reserved, MIB);

    // The budget is a hard cap.
    let _fill_1 = memory_management.reserve(MIB).unwrap();
    let _fill_2 = memory_management.reserve(500 * 1024).unwrap();
    assert!(memory_management.reserve(MIB).is_err());
    assert_eq!(memory_management.memory_usage().bytes_reserved, 2 * MIB);
}

#[test]
fn capped_pool_respects_max_slice_size() {
    // A small hard-capped pool routing only tiny allocations (e.g. kernel
    // metadata), followed by a big arena. Allocations near the small pool's
    // *page* size must not be pulled into it by the low-fragmentation
    // heuristic — that would drain the small pool's budget with strays the
    // cap was never sized for (seen on wgpu: 8 MiB upload staging chunks
    // exhausting an 8 MiB-page metadata pool).
    let pools = MemoryPoolsConfig::Explicit(vec![
        MemoryPoolConfig::Sliced {
            page_size: MemorySize(MIB),
            max_slice_size: Some(MemorySize(64 * 1024)),
            max_pool_size: Some(MemorySize(2 * MIB)),
            dealloc_period: None,
        },
        MemoryPoolConfig::Sliced {
            page_size: MemorySize(16 * MIB),
            max_slice_size: None,
            max_pool_size: Some(MemorySize(32 * MIB)),
            dealloc_period: None,
        },
    ]);
    let resolved = MemoryConfiguration::default()
        .resolve(Some(&pools), &props())
        .unwrap();
    let mut memory_management = MemoryManagement::from_configuration(
        BytesStorage::default(),
        &props(),
        resolved,
        Arc::new(ServerLogger::default()),
        MemoryManagementOptions::new("Main GPU Memory"),
    );

    // Page-sized allocations in the small pool's range go to the arena: even
    // many of them (more than the small pool's whole budget) must succeed.
    let strays: Vec<_> = (0..4)
        .map(|_| memory_management.reserve(MIB).unwrap())
        .collect();
    // Nothing landed in the small pool — its pages were never allocated.
    assert_eq!(
        memory_management.memory_usage().bytes_reserved,
        16 * MIB,
        "strays must land in the arena, not the capped small pool"
    );

    // Small allocations still route to the small pool.
    let _tiny = memory_management.reserve(4096).unwrap();
    assert_eq!(memory_management.memory_usage().bytes_reserved, 17 * MIB);
    drop(strays);
}

#[test]
fn configure_rebuilds_pools_in_place() {
    let resolved = MemoryConfiguration::default()
        .resolve(Some(&sliced(MIB, 2)), &props())
        .unwrap();
    let mut memory_management = MemoryManagement::from_configuration(
        BytesStorage::default(),
        &props(),
        resolved,
        Arc::new(ServerLogger::default()),
        MemoryManagementOptions::new("Main GPU Memory"),
    );

    // While an allocation is live, the rebuild is refused and the old layout
    // (2 × 1 MiB cap) stays in force.
    let live = memory_management.reserve(MIB).unwrap();
    let bigger = MemoryConfiguration::default()
        .resolve(Some(&sliced(4 * MIB, 2)), &props())
        .unwrap();
    assert!(
        matches!(
            memory_management.install_pools(bigger.clone(), &props()),
            Err(InstallMemoryPoolsError::PoolsInUse { bytes_in_use }) if bytes_in_use > 0
        ),
        "the refusal names the live bytes that caused it"
    );
    assert!(memory_management.reserve(2 * MIB).is_err());

    // At a quiescent point the rebuild goes through, and the new layout
    // serves what the old cap refused.
    drop(live);
    memory_management.install_pools(bigger, &props()).unwrap();
    let _large = memory_management.reserve(2 * MIB).unwrap();
}

fn manage(pools: &MemoryPoolsConfig) -> MemoryManagement<BytesStorage> {
    let resolved = MemoryConfiguration::default()
        .resolve(Some(pools), &props())
        .unwrap();
    MemoryManagement::from_configuration(
        BytesStorage::default(),
        &props(),
        resolved,
        Arc::new(ServerLogger::default()),
        MemoryManagementOptions::new("Main GPU Memory"),
    )
}

/// A plan measured from one run of a stream fits that stream when it is
/// replayed: measure against a growable arena, cap the arena at the observed
/// high-water, replay, and the peak is unchanged.
///
/// This is the property the whole report exists to provide. It holds only
/// because pool placement is deterministic — if first-fit ever became
/// order-dependent or randomized, a capped replay could overflow a cap its own
/// measurement produced.
#[test]
fn measured_plan_cycle() {
    // Phase 1 runs the stream against the growable arena; in production this
    // happens under a `DryRun`, where it costs no compute.
    let growable = MemoryPoolsConfig::Explicit(vec![MemoryPoolConfig::Sliced {
        page_size: MemorySize(MIB),
        max_slice_size: None,
        max_pool_size: None,
        dealloc_period: None,
    }]);
    let mut memory_management = manage(&growable);

    // A workload-shaped stream: long-lived buffers overlapping transients,
    // with a reuse (the 900 KiB fits where the dropped 600 KiB was, after
    // coalescing with the page remainder).
    let workload = |memory_management: &mut MemoryManagement<BytesStorage>| {
        let a = memory_management.reserve(600 * 1024).unwrap();
        let b = memory_management.reserve(600 * 1024).unwrap();
        drop(a);
        let c = memory_management.reserve(900 * 1024).unwrap();
        drop(b);
        drop(c);
    };
    workload(&mut memory_management);

    let report = memory_management.memory_report();
    let arena = &report.dynamic[0];
    let MemoryPoolKind::Sliced { page_size, .. } = arena.kind else {
        panic!("the arena is a sliced pool");
    };
    assert_eq!(arena.largest_alloc, 900 * 1024);
    assert_eq!(arena.pages_peak, 2, "two pages while a, b overlap");

    // Cap the arena at the measured high-water.
    let capped = MemoryPoolsConfig::Explicit(vec![MemoryPoolConfig::Sliced {
        page_size: MemorySize(page_size),
        max_slice_size: None,
        max_pool_size: Some(MemorySize(page_size * arena.pages_peak)),
        dealloc_period: None,
    }]);
    let resolved = MemoryConfiguration::default()
        .resolve(Some(&capped), &props())
        .unwrap();
    memory_management.cleanup(true);
    memory_management.install_pools(resolved, &props()).unwrap();

    // The replayed stream fits the cap without growing past the plan.
    workload(&mut memory_management);
    let replayed = memory_management.memory_report();
    assert_eq!(replayed.dynamic[0].pages_peak, arena.pages_peak);
}

/// A cap is a plan, not a budget: an allocation a measured arena has no room
/// for spills to the growable pool behind it rather than failing.
///
/// A plan that turns out to be short should cost memory, not kill the
/// workload — a stream the measuring run never saw is a normal thing to meet
/// in production. Where a cap really is a hard budget, configure no pool
/// behind it and a full pool still errors
/// (`programmatic_pools_override_runtime_default` covers that side).
#[test]
fn full_capped_pool_spills_to_tail() {
    let pools = MemoryPoolsConfig::Explicit(vec![
        MemoryPoolConfig::Sliced {
            page_size: MemorySize(MIB),
            max_slice_size: None,
            max_pool_size: Some(MemorySize(MIB)),
            dealloc_period: None,
        },
        MemoryPoolConfig::Sliced {
            page_size: MemorySize(4 * MIB),
            max_slice_size: None,
            max_pool_size: None,
            dealloc_period: None,
        },
    ]);
    let mut memory_management = manage(&pools);

    let _planned = memory_management.reserve(MIB).unwrap();
    let _off_plan = memory_management.reserve(MIB).unwrap();

    let report = memory_management.memory_report();
    assert_eq!(report.dynamic[0].pages_peak, 1, "the arena stayed capped");
    assert_eq!(report.dynamic[1].pages_peak, 1, "the tail caught the spill");
}

/// A dry run needs no allocation routing of its own, not even for the tuning
/// it exists to provoke: every allocation is carved lazily, and resolution is
/// what installs backing. A measurement resolves because it executes, so it
/// gets its memory; a skipped launch never resolves, so its reservation costs
/// nothing.
///
/// That symmetry is the whole reason a measurement needs no pool of its own.
#[test]
#[serial_test::serial]
fn a_measurement_maps_only_what_it_resolves() {
    let growable = MemoryPoolsConfig::Explicit(vec![MemoryPoolConfig::Sliced {
        page_size: MemorySize(MIB),
        max_slice_size: None,
        max_pool_size: None,
        dealloc_period: None,
    }]);
    let mut memory_management = manage(&growable);

    let dry_run = DryRun::new();

    // A workload reservation, and a measurement's scratch beside it. At
    // 600 KiB each they cannot share a 1 MiB page, so the two are separable.
    let workload = memory_management.reserve(600 * 1024).unwrap();
    let scratch = {
        let _measurement = RealRun::new();
        memory_management.reserve(600 * 1024).unwrap()
    };

    let report = memory_management.memory_report();
    assert_eq!(report.dynamic[0].pages, 2, "{report:?}");
    assert_eq!(
        report.dynamic[0].pages_unmapped, 2,
        "neither is backed until something resolves it: {report:?}"
    );

    // The measurement executes, so it resolves — and that, not its provenance,
    // is what maps it.
    memory_management
        .get_storage(scratch.clone().binding())
        .unwrap();

    let report = memory_management.memory_report();
    assert_eq!(
        report.dynamic[0].pages_unmapped, 1,
        "the measurement's page is backed and the workload's is not: {report:?}"
    );

    drop(workload);
    drop(scratch);
    drop(dry_run);
}

/// The phase-2 laziness invariant: a dry run's workload reservations carve
/// pages, count toward every high-water mark, and cost no device memory —
/// backing is installed on demand the first time a binding is resolved into
/// something that executes (mapped ⊆ reserved, worst case = eager).
#[test]
#[serial_test::serial]
fn dry_run_reservations_stay_unmapped_until_resolved() {
    let growable = MemoryPoolsConfig::Explicit(vec![MemoryPoolConfig::Sliced {
        page_size: MemorySize(MIB),
        max_slice_size: None,
        max_pool_size: None,
        dealloc_period: None,
    }]);
    let mut memory_management = manage(&growable);

    let dry_run = DryRun::new();
    let reserved = memory_management.reserve(600 * 1024).unwrap();

    let report = memory_management.memory_report();
    assert_eq!(report.dynamic[0].pages, 1, "the page was carved");
    assert_eq!(
        report.dynamic[0].pages_peak, 1,
        "and counts toward the plan"
    );
    assert_eq!(
        report.dynamic[0].pages_unmapped, 1,
        "but has no device backing: {report:?}"
    );

    // Resolution is the materialization point: after it the handle refers to
    // real memory and the page reads as mapped.
    let storage = memory_management
        .get_storage(reserved.clone().binding())
        .unwrap();
    assert_eq!(storage.size(), 600 * 1024);
    assert_eq!(
        memory_management.memory_report().dynamic[0].pages_unmapped,
        0,
        "resolution installed the backing"
    );

    drop(reserved);
    drop(dry_run);

    // Unmapped or mapped, cleanup must not corrupt anything.
    memory_management.cleanup(true);
    assert_eq!(memory_management.memory_report().dynamic[0].pages, 0);
}

/// Persistent reservations under a dry run are lazy too — a scratch session's
/// KV cache is the big one — and materialize per slice when a measurement
/// actually touches them.
#[test]
#[serial_test::serial]
fn dry_run_persistent_reservations_stay_unmapped() {
    let growable = MemoryPoolsConfig::Explicit(vec![MemoryPoolConfig::Sliced {
        page_size: MemorySize(MIB),
        max_slice_size: None,
        max_pool_size: None,
        dealloc_period: None,
    }]);
    let mut memory_management = manage(&growable);
    memory_management.mode(MemoryAllocationMode::Persistent);

    let dry_run = DryRun::new();
    let kv = memory_management.reserve(512 * 1024).unwrap();

    let report = memory_management.memory_report();
    assert_eq!(report.persistent.pages, 1);
    assert_eq!(
        report.persistent.pages_unmapped, 1,
        "a dry-run persistent slice has no backing yet: {report:?}"
    );

    let storage = memory_management.get_storage(kv.clone().binding()).unwrap();
    assert_eq!(storage.size(), 512 * 1024);
    assert_eq!(
        memory_management.memory_report().persistent.pages_unmapped,
        0
    );

    drop(kv);
    drop(dry_run);
    memory_management.mode(MemoryAllocationMode::Auto);
}

/// Tuning scratch counts toward the marks like any other allocation, so a plan
/// is read from a *second* pass: the tune caches are warm by then, and
/// rebuilding the pools resets the marks the first pass left.
///
/// Read from the first pass instead, the plan is sized for scratch that will
/// never be allocated again — over-provisioning to replace the padding the
/// report exists to remove.
#[test]
#[serial_test::serial]
fn a_warm_second_pass_measures_the_workload_alone() {
    let growable = MemoryPoolsConfig::Explicit(vec![MemoryPoolConfig::Sliced {
        page_size: MemorySize(MIB),
        max_slice_size: None,
        max_pool_size: None,
        dealloc_period: None,
    }]);
    let mut memory_management = manage(&growable);

    let dry_run = DryRun::new();

    // Pass 1: the workload, with a tuning pass allocating scratch while a
    // workload buffer is live — which is when tuning actually happens.
    let live = memory_management.reserve(600 * 1024).unwrap();
    {
        let _measurement = RealRun::new();
        let scratch = memory_management.reserve(600 * 1024).unwrap();
        drop(scratch);
    }
    drop(live);
    assert_eq!(
        memory_management.memory_report().dynamic[0].pages_peak,
        2,
        "the scratch is in the marks: it is an allocation like any other"
    );

    // Rebuilding the pools resets the marks. The tune caches are untouched by
    // it, so the second pass allocates nothing for them.
    let resolved = MemoryConfiguration::default()
        .resolve(Some(&growable), &props())
        .unwrap();
    memory_management.install_pools(resolved, &props()).unwrap();

    // Pass 2: the same workload, no tuning.
    let live = memory_management.reserve(600 * 1024).unwrap();
    drop(live);

    let report = memory_management.memory_report();
    assert_eq!(
        report.dynamic[0].pages_peak, 1,
        "the plan is the workload's own high-water: {report:?}"
    );

    drop(dry_run);
}

/// The direct pool's reason to exist: it wastes only alignment padding, where
/// a sliced arena wastes the remainder of every page it carves.
///
/// That is the whole trade — a device allocation per distinct size, bought
/// with the padding it removes — so a layout that does not actually remove the
/// padding is not worth its allocation cost.
#[test]
fn direct_pool_pads_only_to_alignment() {
    let mut memory_management = manage(&MemoryPoolsConfig::Explicit(vec![
        MemoryPoolConfig::Direct { reclaim_at: None },
    ]));

    // A size that is neither page- nor bucket-shaped: a sliced pool would
    // reserve a whole page for it.
    let odd = 700 * 1024 + 17;
    let _live = memory_management.reserve(odd).unwrap();

    let report = memory_management.memory_report();
    assert!(matches!(report.dynamic[0].kind, MemoryPoolKind::Direct));
    assert_eq!(report.dynamic[0].usage.bytes_in_use, odd);
    assert_eq!(
        report.dynamic[0].usage.bytes_reserved,
        odd.next_multiple_of(props().alignment),
        "nothing is reserved beyond what alignment demands: {report:?}"
    );
}

/// Below the ceiling the pool holds freed slices and reuses them by exact
/// size, so a loop over one shape allocates once.
///
/// This is what keeps the pool from distorting an autotune measurement: a
/// benchmark's iterations reuse rather than paying for driver traffic the real
/// workload never pays — and paying unequally across candidates by how much
/// each allocates, which is measurement error that looks like a result.
#[test]
fn direct_pool_reuses_below_the_ceiling() {
    let mut memory_management = manage(&MemoryPoolsConfig::Explicit(vec![
        MemoryPoolConfig::Direct {
            reclaim_at: Some(MemorySize(8 * MIB)),
        },
    ]));

    for _ in 0..4 {
        let scratch = memory_management.reserve(300 * 1024).unwrap();
        drop(scratch);
    }

    let report = memory_management.memory_report();
    assert_eq!(
        report.dynamic[0].pages_peak, 1,
        "four iterations of one shape allocated once: {report:?}"
    );
}

/// Crossing the ceiling is the only thing that returns memory to the driver,
/// and it returns just enough for the allocation that crossed it.
///
/// Reclaiming everything would throw away reuse the pool has not been asked to
/// give up; reclaiming nothing would make the ceiling advisory.
#[test]
fn direct_pool_reclaims_at_the_ceiling() {
    let mut memory_management = manage(&MemoryPoolsConfig::Explicit(vec![
        MemoryPoolConfig::Direct {
            reclaim_at: Some(MemorySize(7 * MIB)),
        },
    ]));

    // 1 + 2 + 3 MiB, all freed and all distinct sizes, so none is reusable for
    // a fourth: 6 MiB held under a 7 MiB ceiling.
    for size in [MIB, 2 * MIB, 3 * MIB] {
        let held = memory_management.reserve(size).unwrap();
        drop(held);
    }
    let report = memory_management.memory_report();
    assert_eq!(
        report.dynamic[0].pages, 3,
        "held below the ceiling: {report:?}"
    );

    // A 4 MiB allocation needs 3 MiB back. Visiting in index order, releasing
    // the 1 MiB and 2 MiB slices is enough — the 3 MiB one stays reusable.
    let _crossed = memory_management.reserve(4 * MIB).unwrap();
    let report = memory_management.memory_report();
    assert_eq!(
        report.dynamic[0].pages, 2,
        "just enough was released, not everything free: {report:?}"
    );
    assert_eq!(
        report.dynamic[0].usage.bytes_reserved,
        7 * MIB,
        "the kept 3 MiB slice plus the new 4 MiB one: {report:?}"
    );
}

/// An explicit cleanup returns every free slice regardless of the ceiling: the
/// caller is stating that reuse is worth less than the memory right now.
#[test]
fn direct_pool_cleanup_releases_everything_free() {
    let mut memory_management = manage(&MemoryPoolsConfig::Explicit(vec![
        MemoryPoolConfig::Direct {
            reclaim_at: Some(MemorySize(64 * MIB)),
        },
    ]));

    let live = memory_management.reserve(MIB).unwrap();
    let freed = memory_management.reserve(2 * MIB).unwrap();
    drop(freed);

    memory_management.cleanup(true);
    let report = memory_management.memory_report();
    assert_eq!(
        report.dynamic[0].pages, 1,
        "only the live slice survives: {report:?}"
    );
    drop(live);
}
