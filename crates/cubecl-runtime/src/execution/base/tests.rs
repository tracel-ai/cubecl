use super::super::{ProcessModeOverride, StatisticsCollector, StreamModeOverride};
use super::*;
use cubecl_common::device::{DeviceId, ServiceId};
use cubecl_environment::stream::StreamId;
// `serial_test`'s macro expands to `vec!`, which a `no_std` crate has to
// bring in itself.
use alloc::vec;

/// Stream `value` of the service of device `index`.
fn stream(index: u16, value: u64) -> ServiceStream {
    ServiceStream {
        service: ServiceId::of::<()>(DeviceId {
            type_id: 0,
            index_id: index,
        }),
        stream: StreamId { value },
    }
}

/// The verdict of a launch on `stream` under an override of `mode`.
fn under(mode: ProcessMode, verdict: impl FnOnce() -> LaunchMode) -> LaunchMode {
    let collector = StatisticsCollector::new();
    let _override = ProcessModeOverride::new(mode, &collector);
    verdict()
}

#[test]
#[serial_test::serial]
fn a_launch_runs_with_no_override_open() {
    assert_eq!(LaunchMode::new(stream(0, 7)), LaunchMode::Execute);
}

#[test]
#[serial_test::serial]
fn a_launch_queues_under_compile_only() {
    let mode = under(ProcessMode::CompileOnly, || LaunchMode::new(stream(0, 7)));
    assert_eq!(mode, LaunchMode::Queue);
}

#[test]
#[serial_test::serial]
fn a_launch_compiles_under_compile_and_autotune() {
    let mode = under(ProcessMode::CompileAndAutotune, || {
        LaunchMode::new(stream(0, 7))
    });
    assert_eq!(mode, LaunchMode::Compile);
}

#[test]
#[serial_test::serial]
fn a_launch_runs_under_an_execute_override() {
    let mode = under(ProcessMode::Execute, || LaunchMode::new(stream(0, 7)));
    assert_eq!(mode, LaunchMode::Execute);
}

/// A stream discarding its launches while the process executes queues them.
#[test]
#[serial_test::serial]
fn a_discarding_stream_queues_while_the_process_executes() {
    let discarding = StreamModeOverride::on_stream(StreamMode::Discard, stream(0, 7));
    assert_eq!(LaunchMode::new(stream(0, 7)), LaunchMode::Queue);
    core::mem::drop(discarding);
}

/// A launch of a planning stream resolves: the recorded run then finds
/// everything the launch path allocates already in the pools.
#[test]
#[serial_test::serial]
fn a_planning_stream_resolves() {
    assert_eq!(LaunchMode::planning(stream(0, 9)), LaunchMode::Resolve);
}

/// What a tune measuring inside the planning run opens.
#[test]
#[serial_test::serial]
fn a_planning_stream_runs_where_an_override_runs_it() {
    let measuring = StreamModeOverride::on_stream(StreamMode::Execute, stream(0, 9));
    assert_eq!(LaunchMode::planning(stream(0, 9)), LaunchMode::Execute);
    core::mem::drop(measuring);
}

#[test]
#[serial_test::serial]
fn a_planning_stream_resolves_where_an_override_discards_it() {
    let gathering = StreamModeOverride::on_stream(StreamMode::Discard, stream(0, 9));
    assert_eq!(LaunchMode::planning(stream(0, 9)), LaunchMode::Resolve);
    core::mem::drop(gathering);
}

/// Under a process mode that discards every launch, a planning stream's
/// launches follow that mode: they compile in its batches.
#[test]
#[serial_test::serial]
fn a_planning_stream_queues_under_compile_only() {
    let mode = under(ProcessMode::CompileOnly, || {
        LaunchMode::planning(stream(0, 10))
    });
    assert_eq!(mode, LaunchMode::Queue);
}

#[test]
#[serial_test::serial]
fn a_planning_stream_compiles_under_compile_and_autotune() {
    let mode = under(ProcessMode::CompileAndAutotune, || {
        LaunchMode::planning(stream(0, 10))
    });
    assert_eq!(mode, LaunchMode::Compile);
}

/// A stream's mode holds for that stream of that device alone.
#[test]
#[serial_test::serial]
fn a_stream_mode_holds_for_its_stream() {
    let mode = under(ProcessMode::CompileAndAutotune, || {
        let _measuring = StreamModeOverride::on_stream(StreamMode::Execute, stream(0, 1));
        LaunchMode::new(stream(0, 1))
    });
    assert_eq!(mode, LaunchMode::Execute);
}

#[test]
#[serial_test::serial]
fn a_stream_mode_leaves_another_stream_of_its_device_alone() {
    let mode = under(ProcessMode::CompileAndAutotune, || {
        let _measuring = StreamModeOverride::on_stream(StreamMode::Execute, stream(0, 1));
        LaunchMode::new(stream(0, 2))
    });
    assert_eq!(mode, LaunchMode::Compile);
}

#[test]
#[serial_test::serial]
fn a_stream_mode_leaves_the_same_stream_of_another_device_alone() {
    let mode = under(ProcessMode::CompileAndAutotune, || {
        let _measuring = StreamModeOverride::on_stream(StreamMode::Execute, stream(0, 1));
        LaunchMode::new(stream(1, 1))
    });
    assert_eq!(mode, LaunchMode::Compile);
}

#[test]
#[serial_test::serial]
fn the_newest_stream_mode_of_a_stream_decides() {
    let mode = under(ProcessMode::CompileAndAutotune, || {
        let _measuring = StreamModeOverride::on_stream(StreamMode::Execute, stream(0, 1));
        let _nested = StreamModeOverride::on_stream(StreamMode::Discard, stream(0, 1));
        LaunchMode::new(stream(0, 1))
    });
    assert_eq!(mode, LaunchMode::Compile);
}

/// Overrides of one stream drop in any order: the older dropping first
/// leaves the newer deciding.
#[test]
#[serial_test::serial]
fn stream_modes_drop_in_any_order() {
    let mode = under(ProcessMode::CompileAndAutotune, || {
        let measuring = StreamModeOverride::on_stream(StreamMode::Execute, stream(0, 1));
        let _nested = StreamModeOverride::on_stream(StreamMode::Discard, stream(0, 1));
        core::mem::drop(measuring);
        LaunchMode::new(stream(0, 1))
    });
    assert_eq!(mode, LaunchMode::Compile);
}
