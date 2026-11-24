use super::super::statistics::{KernelHandling, KernelOutcome, TuneOutcome, TunePick};
use super::super::{CompilationStatistics, KernelRegistration, TuneRegistration};
use super::*;
// `serial_test`'s macro expands to `vec!`, which a `no_std` crate has to
// bring in itself.
use alloc::vec;
use alloc::vec::Vec;

/// One process mode is open at a time: an override of another refuses to open
/// beside it.
#[test]
#[serial_test::serial]
#[should_panic(
    expected = "an override of CompileOnly cannot open while one of CompileAndAutotune is"
)]
fn process_modes_do_not_overlap() {
    let collector = StatisticsCollector::new();
    let opened = ProcessModeOverride::new(ProcessMode::CompileAndAutotune, &collector);
    let refused = ProcessModeOverride::new(ProcessMode::CompileOnly, &collector);
    core::mem::drop(opened);
    core::mem::drop(refused);
}

/// The work under an override counts to one collector: another's
/// refuses to open beside it, whatever its process mode.
#[test]
#[serial_test::serial]
#[should_panic(expected = "cannot open while one of another collector is")]
fn two_collectors_do_not_overlap() {
    let (first, second) = (StatisticsCollector::new(), StatisticsCollector::new());
    let opened = ProcessModeOverride::new(ProcessMode::CompileOnly, &first);
    let refused = ProcessModeOverride::new(ProcessMode::CompileOnly, &second);
    core::mem::drop(opened);
    core::mem::drop(refused);
}

/// A refused open leaves the open override as it was.
#[test]
#[serial_test::serial]
fn a_refused_open_leaves_the_open_process_mode() {
    let collector = StatisticsCollector::new();
    let tune = ProcessModeOverride::new(ProcessMode::CompileAndAutotune, &collector);
    let refused =
        std::panic::catch_unwind(|| ProcessModeOverride::new(ProcessMode::CompileOnly, &collector));
    assert!(refused.is_err());
    assert_eq!(ProcessMode::current(), ProcessMode::CompileAndAutotune);
    core::mem::drop(tune);
    assert_eq!(ProcessMode::current(), ProcessMode::Execute);
}

/// Overlapping overrides of one process mode and collector compose: the last
/// out closes it.
#[test]
#[serial_test::serial]
fn overrides_nest() {
    let collector = StatisticsCollector::new();
    {
        let outer = ProcessModeOverride::new(ProcessMode::CompileOnly, &collector);
        {
            let inner = ProcessModeOverride::new(ProcessMode::CompileOnly, &collector);
            core::mem::drop(inner);
        }
        assert_eq!(
            ProcessMode::current(),
            ProcessMode::CompileOnly,
            "the outer is in force"
        );
        core::mem::drop(outer);
    }
    assert_eq!(ProcessMode::current(), ProcessMode::Execute);
    assert!(ProcessModeOverride::active_recorder().is_none());
}

/// What an override triggers counts to its collector, over every
/// override of it — an `Execute` one included — and a registration made
/// under one settles there after it closes; what happens to settled work
/// counts once; a reader reads the same counts.
#[test]
#[serial_test::serial]
fn work_counts_to_the_collector_of_the_open_override() {
    assert!(
        KernelRegistration::register().is_none(),
        "nothing counts outside an override"
    );
    let collector = StatisticsCollector::new();
    let reader = collector.reader();

    let compile = ProcessModeOverride::new(ProcessMode::CompileOnly, &collector);
    let queued: Vec<KernelRegistration> = (0..4)
        .filter_map(|_| KernelRegistration::register())
        .collect();
    let gathered = TuneRegistration::register().expect("an override is open");
    core::mem::drop(compile);

    // Kernels queued in one override compile in the next.
    let tune = ProcessModeOverride::new(ProcessMode::CompileAndAutotune, &collector);
    let mut queued = queued.into_iter();
    let mut compiled = queued.next().unwrap().settle(KernelOutcome::Compiled);
    compiled.record(KernelHandling::Refused);
    compiled.record(KernelHandling::Refused);
    let mut loaded = queued.next().unwrap().settle(KernelOutcome::Loaded);
    loaded.record(KernelHandling::Stored);
    let _failed = queued.next().unwrap().settle(KernelOutcome::Failed);
    let mut measured = gathered.settle(TuneOutcome::Measured);
    measured.record(TunePick::Persisted);

    let statistics = collector.statistics();
    assert_eq!(
        statistics.compilation,
        CompilationStatistics {
            registered: 4,
            compiled: 1,
            loaded: 1,
            failed: 1,
            abandoned: 0,
            refused: 1,
            stored: 1,
        },
        "a refusal told twice counts once"
    );
    assert_eq!(
        statistics.autotune.settled(),
        statistics.autotune.registered
    );
    assert_eq!(statistics.autotune.persisted, 1);
    assert_eq!(reader.statistics(), statistics);
    core::mem::drop(tune);

    let execute = ProcessModeOverride::new(ProcessMode::Execute, &collector);
    let _measured = TuneRegistration::register()
        .expect("an execute override counts too")
        .settle(TuneOutcome::Measured);
    core::mem::drop(execute);
    assert_eq!(collector.statistics().autotune.measured, 2);
    // The fourth kernel, never settled, is abandoned as it drops.
    drop(queued);
    assert_eq!(collector.statistics().compilation.abandoned, 1);
}

/// A registration dropped before it settled — its batch or its tune
/// panicked — settles as abandoned.
#[test]
#[serial_test::serial]
fn a_dropped_registration_settles_as_abandoned() {
    let collector = StatisticsCollector::new();
    let compile = ProcessModeOverride::new(ProcessMode::CompileOnly, &collector);
    drop(KernelRegistration::register());
    drop(TuneRegistration::register());
    let statistics = collector.statistics();
    assert_eq!(
        (
            statistics.compilation.abandoned,
            statistics.compilation.settled()
        ),
        (1, 1)
    );
    assert_eq!(
        (statistics.autotune.abandoned, statistics.autotune.settled()),
        (1, 1)
    );
    core::mem::drop(compile);
}
