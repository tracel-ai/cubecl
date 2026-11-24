use super::super::{
    KernelOutcome, KernelRegistration, StatisticsRecorder, TuneOutcome, TuneRegistration,
};
use super::*;
// `serial_test`'s macro expands to `vec!`, which a `no_std` crate has to
// bring in itself.
use alloc::vec;

/// One process mode is open at a time: an override of another refuses to open
/// beside it.
#[test]
#[serial_test::serial]
#[should_panic(
    expected = "an override of CompileOnly cannot open while one of CompileAndAutotune is"
)]
fn process_modes_do_not_overlap() {
    let collector = StatisticsCollector::new();
    let _opened = ProcessModeOverride::new(ProcessMode::CompileAndAutotune, &collector);
    let _refused = ProcessModeOverride::new(ProcessMode::CompileOnly, &collector);
}

/// The work under an override counts to one collector: another's
/// refuses to open beside it, whatever its process mode.
#[test]
#[serial_test::serial]
#[should_panic(expected = "cannot open while one of another collector is")]
fn two_collectors_do_not_overlap() {
    let (first, second) = (StatisticsCollector::new(), StatisticsCollector::new());
    let _opened = ProcessModeOverride::new(ProcessMode::CompileOnly, &first);
    let _refused = ProcessModeOverride::new(ProcessMode::CompileOnly, &second);
}

#[test]
#[serial_test::serial]
fn a_refused_open_leaves_the_open_process_mode() {
    let collector = StatisticsCollector::new();
    let tune = ProcessModeOverride::new(ProcessMode::CompileAndAutotune, &collector);
    let refused =
        std::panic::catch_unwind(|| ProcessModeOverride::new(ProcessMode::CompileOnly, &collector));
    assert_eq!(
        (refused.is_err(), ProcessMode::current()),
        (true, ProcessMode::CompileAndAutotune)
    );
    core::mem::drop(tune);
}

#[test]
#[serial_test::serial]
fn no_override_open_executes() {
    assert_eq!(ProcessMode::current(), ProcessMode::Execute);
}

#[test]
#[serial_test::serial]
fn an_override_applies_its_process_mode() {
    let collector = StatisticsCollector::new();
    let compile = ProcessModeOverride::new(ProcessMode::CompileOnly, &collector);
    assert_eq!(ProcessMode::current(), ProcessMode::CompileOnly);
    core::mem::drop(compile);
}

/// Overlapping overrides of one process mode and collector compose: the
/// outer is still in force once the inner drops.
#[test]
#[serial_test::serial]
fn an_override_joined_stays_open_until_its_last_guard_drops() {
    let collector = StatisticsCollector::new();
    let outer = ProcessModeOverride::new(ProcessMode::CompileOnly, &collector);
    let inner = ProcessModeOverride::new(ProcessMode::CompileOnly, &collector);
    core::mem::drop(inner);
    assert_eq!(ProcessMode::current(), ProcessMode::CompileOnly);
    core::mem::drop(outer);
}

#[test]
#[serial_test::serial]
fn the_last_guard_out_closes_the_override_and_its_collector() {
    let collector = StatisticsCollector::new();
    let outer = ProcessModeOverride::new(ProcessMode::CompileOnly, &collector);
    let inner = ProcessModeOverride::new(ProcessMode::CompileOnly, &collector);
    core::mem::drop(inner);
    core::mem::drop(outer);
    assert_eq!(
        (
            ProcessMode::current(),
            StatisticsRecorder::active().is_none()
        ),
        (ProcessMode::Execute, true)
    );
}

#[test]
#[serial_test::serial]
fn nothing_counts_outside_an_override() {
    assert!(KernelRegistration::register().is_none());
}

#[test]
#[serial_test::serial]
fn work_counts_to_the_collector_of_the_open_override() {
    let collector = StatisticsCollector::new();
    let compile = ProcessModeOverride::new(ProcessMode::CompileOnly, &collector);
    let registered = KernelRegistration::register();
    core::mem::drop(compile);
    assert_eq!(collector.statistics().compilation.registered_count, 1);
    core::mem::drop(registered);
}

/// An override of [`ProcessMode::Execute`] counts too: it tallies what a
/// real run compiles and tunes.
#[test]
#[serial_test::serial]
fn an_execute_override_counts_too() {
    let collector = StatisticsCollector::new();
    let execute = ProcessModeOverride::new(ProcessMode::Execute, &collector);
    let registered = TuneRegistration::register();
    core::mem::drop(execute);
    assert_eq!(collector.statistics().autotune.registered_count, 1);
    core::mem::drop(registered);
}

/// A kernel queued in one override compiles in the next, and counts to the
/// collector it was registered with.
#[test]
#[serial_test::serial]
fn a_registration_settles_where_it_registered_after_its_override_closed() {
    let collector = StatisticsCollector::new();
    let compile = ProcessModeOverride::new(ProcessMode::CompileOnly, &collector);
    let queued =
        KernelRegistration::register().expect("a kernel queued under an override should register");
    core::mem::drop(compile);

    let settled = queued.settle(KernelOutcome::Compiled);

    assert_eq!(collector.statistics().compilation.compiled_count, 1);
    core::mem::drop(settled);
}

/// A tune gathered under one override and measured under the next counts
/// once, to the collector it was gathered with.
#[test]
#[serial_test::serial]
fn a_tune_gathered_and_measured_across_overrides_settles_once() {
    let collector = StatisticsCollector::new();
    let compile = ProcessModeOverride::new(ProcessMode::CompileOnly, &collector);
    let gathered =
        TuneRegistration::register().expect("a tune gathered under an override should register");
    core::mem::drop(compile);
    let tune = ProcessModeOverride::new(ProcessMode::CompileAndAutotune, &collector);

    let measured = gathered.settle(TuneOutcome::Measured);

    core::mem::drop(tune);
    let statistics = collector.statistics().autotune;
    assert_eq!(
        (statistics.registered_count, statistics.measured_count),
        (1, 1)
    );
    core::mem::drop(measured);
}
