use super::super::{
    KernelOutcome, KernelRegistration, ProcessMode, ProcessModeOverride, StatisticsCollector,
};
use super::*;
// `serial_test`'s macro expands to `vec!`, which a `no_std` crate has to
// bring in itself.
use alloc::vec;

/// A launch issued under one override counts there when it runs after
/// that override closed — even while another collector's is open — and
/// one issued under none counts nowhere.
#[test]
#[serial_test::serial]
fn a_launch_counts_where_it_was_issued() {
    let issuing = StatisticsCollector::new();
    let gathering = ProcessModeOverride::new(ProcessMode::CompileOnly, &issuing);
    let issued = IssuedRecorder::new();
    core::mem::drop(gathering);

    let later = StatisticsCollector::new();
    let tuning = ProcessModeOverride::new(ProcessMode::CompileAndAutotune, &later);
    let registered = issued.apply(KernelRegistration::register);
    let outside = IssuedRecorder { recorder: None }.apply(KernelRegistration::register);
    core::mem::drop(tuning);

    let settled = registered
        .expect("issued under an override")
        .settle(KernelOutcome::Compiled);
    assert!(outside.is_none(), "issued under none, it counts nowhere");
    assert_eq!(issuing.statistics().compilation.compiled, 1);
    assert_eq!(later.statistics().compilation.registered, 0);
    core::mem::drop(settled);
}
