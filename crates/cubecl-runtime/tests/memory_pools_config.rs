//! End-to-end test of the programmatic pool layout: the whole path from a
//! [`MemoryPoolsConfig`] payload through [`MemoryConfiguration::resolve`] to
//! the actual pool behavior, plus the in-place rebuild
//! ([`MemoryManagement::configure`]) that re-sizes the pools between
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
    MemoryAllocationMode, MemoryConfiguration, MemoryManagement,
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
    assert!(!memory_management.configure(bigger.clone(), &props()));
    assert!(memory_management.reserve(2 * MIB).is_err());

    // At a quiescent point the rebuild goes through, and the new layout
    // serves what the old cap refused.
    drop(live);
    assert!(memory_management.configure(bigger, &props()));
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

#[test]
fn measured_plan_cycle() {
    // The whole measured-plan cycle. Phase 1: run the workload's allocation
    // stream against a growable arena (in production this happens under a
    // `DryRun`, where it costs no compute). Phase 2: read the report and cap
    // the arena at the observed high-water. Phase 3: replay the same stream —
    // pool placement is deterministic, so it fits the cap by construction.
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
    assert!(memory_management.configure(resolved, &props()));

    // The replayed stream fits the cap without growing past the plan.
    workload(&mut memory_management);
    let replayed = memory_management.memory_report();
    assert_eq!(replayed.dynamic[0].pages_peak, arena.pages_peak);
}

#[test]
fn full_capped_pool_spills_to_tail() {
    // A measured arena with a growable tail behind it: an allocation the plan
    // has no room for spills to the tail (loudly, in the logs) instead of
    // failing — the escape hatch for a stream the dry run never measured.
    // With no tail, the same situation is an error (covered above in
    // `programmatic_pools_override_runtime_default`).
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

#[test]
#[serial_test::serial]
fn dry_run_scratch_stays_out_of_the_plan() {
    // Inside a dry run, allocations made by a measurement (RealRun) are
    // scratch for the benchmark being timed, not part of the workload's
    // stream: they must not inflate the high-water marks the dry run measures.
    let growable = MemoryPoolsConfig::Explicit(vec![MemoryPoolConfig::Sliced {
        page_size: MemorySize(MIB),
        max_slice_size: None,
        max_pool_size: None,
        dealloc_period: None,
    }]);
    let mut memory_management = manage(&growable);

    let dry_run = DryRun::new();

    // The workload's own allocation lands in the arena as usual.
    let workload_alloc = memory_management.reserve(64 * 1024).unwrap();

    {
        let _measurement = RealRun::new();
        let scratch = memory_management.reserve(300 * 1024).unwrap();
        assert!(
            memory_management
                .get_cursor(scratch.clone().binding())
                .is_ok(),
            "measurement scratch is a normal, usable allocation"
        );
        drop(scratch);
        // Same size again: the exact-fit pool reuses the slice.
        let _scratch = memory_management.reserve(300 * 1024).unwrap();
    }

    let report = memory_management.memory_report();
    assert_eq!(
        report.dynamic[0].pages_peak, 1,
        "only the workload's allocation counts toward the plan"
    );
    let scratch = report.dry_run.expect("the dry-run pool is reported");
    assert_eq!(scratch.pages_peak, 1, "the same-size scratch was reused");
    assert_eq!(scratch.largest_alloc, 300 * 1024);

    // Outside a dry run a measurement allocates normally: the pools recycle
    // its scratch fine, and routing it away would change production behavior.
    drop(dry_run);
    {
        let _measurement = RealRun::new();
        let _normal = memory_management.reserve(512 * 1024).unwrap();
        let report = memory_management.memory_report();
        assert_eq!(
            report.dynamic[0].usage.bytes_in_use,
            (64 + 512) * 1024,
            "the allocation landed in the arena, beside the workload's"
        );
        assert_eq!(
            report
                .dry_run
                .expect("earlier scratch still reported")
                .pages_peak,
            1,
            "nothing new reached the scratch pool"
        );
    }

    drop(workload_alloc);
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

/// A measurement's scratch dies at the first workload allocation after it —
/// as soon as possible, and never *during* a measurement, whose iterations
/// must stay allocation-free on the exact-fit reuse.
#[test]
#[serial_test::serial]
fn measurement_scratch_dies_at_the_next_workload_allocation() {
    let growable = MemoryPoolsConfig::Explicit(vec![MemoryPoolConfig::Sliced {
        page_size: MemorySize(MIB),
        max_slice_size: None,
        max_pool_size: None,
        dealloc_period: None,
    }]);
    let mut memory_management = manage(&growable);

    let dry_run = DryRun::new();
    {
        let _measurement = RealRun::new();
        let scratch = memory_management.reserve(300 * 1024).unwrap();
        drop(scratch);
        // Same size again: reused, not reallocated — the timed loop stays
        // allocation-free.
        let scratch = memory_management.reserve(300 * 1024).unwrap();
        drop(scratch);
    }

    // Parked while the measurement was open…
    let parked = memory_management
        .memory_report()
        .dry_run
        .expect("the scratch pool served the measurement");
    assert!(parked.usage.bytes_reserved > 0, "{parked:?}");
    assert_eq!(parked.pages_peak, 1, "exact-fit reuse, one slice");

    // …and returned to the driver at the next workload reservation.
    let _workload = memory_management.reserve(64 * 1024).unwrap();
    let flushed = memory_management
        .memory_report()
        .dry_run
        .expect("the peak stays recorded");
    assert_eq!(
        flushed.usage.bytes_reserved, 0,
        "the batch's scratch went back to the driver: {flushed:?}"
    );

    drop(dry_run);
}
