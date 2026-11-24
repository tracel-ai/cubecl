use super::super::{
    KernelOutcome, KernelRegistration, ProcessMode, ProcessModeOverride, StatisticsCollector,
};
use super::*;
// `serial_test`'s macro expands to `vec!`, which a `no_std` crate has to
// bring in itself.
use alloc::vec;

/// A launch issued under one override counts there when it runs after
/// that override closed, even while another collector's is open.
#[test]
#[serial_test::serial]
fn a_launch_counts_where_it_was_issued() {
    let issuing = StatisticsCollector::new();
    let gathering = ProcessModeOverride::new(ProcessMode::CompileOnly, &issuing);
    let issued = IssuedLaunch::new(LaunchPurpose::Work);
    core::mem::drop(gathering);
    let later = StatisticsCollector::new();
    let tuning = ProcessModeOverride::new(ProcessMode::CompileAndAutotune, &later);

    let registered = issued.run(KernelRegistration::register);

    core::mem::drop(tuning);
    let settled = registered
        .expect("a launch issued under an override should register")
        .settle(KernelOutcome::Compiled);
    assert_eq!(
        (
            issuing.statistics().compilation.compiled_count,
            later.statistics().compilation.registered_count
        ),
        (1, 0)
    );
    core::mem::drop(settled);
}
