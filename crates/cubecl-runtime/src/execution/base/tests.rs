use super::super::{ProcessModeOverride, StatisticsCollector};
use super::*;
use cubecl_common::device::{DeviceId, ServiceId};
use cubecl_environment::stream::StreamId;
// `serial_test`'s macro expands to `vec!`, which a `no_std` crate has to
// bring in itself.
use alloc::vec;

/// The service of device `index`.
fn device(index: u16) -> ServiceId {
    ServiceId::of::<()>(DeviceId {
        type_id: 0,
        index_id: index,
    })
}

/// Stream `value`.
fn stream(value: u64) -> StreamId {
    StreamId { value }
}

/// The verdict is a table of a stream's mode and the process mode.
#[test]
#[serial_test::serial]
fn a_launch_follows_its_stream_and_the_process_mode() {
    let collector = StatisticsCollector::new();
    let seventh = ServiceStream {
        service: device(0),
        stream: stream(7),
    };
    let mode = || LaunchMode::new(seventh);
    assert_eq!(mode(), LaunchMode::Execute);
    {
        let compile = ProcessModeOverride::new(ProcessMode::CompileOnly, &collector);
        assert_eq!(mode(), LaunchMode::Queue);
        core::mem::drop(compile);
    }
    {
        let tune = ProcessModeOverride::new(ProcessMode::CompileAndAutotune, &collector);
        assert_eq!(mode(), LaunchMode::Compile);
        core::mem::drop(tune);
    }
    {
        // A stream discarding its launches while the process executes
        // queues them.
        let queuing = STREAM_MODES.set(StreamMode::Discard, seventh);
        assert_eq!(mode(), LaunchMode::Queue);
        core::mem::drop(queuing);
    }
    let execute = ProcessModeOverride::new(ProcessMode::Execute, &collector);
    assert_eq!(mode(), LaunchMode::Execute);
    core::mem::drop(execute);
}

/// A launch of a planning stream resolves, unless an override runs it: what a tune
/// measuring inside the planning run opens.
#[test]
#[serial_test::serial]
fn a_planning_stream_resolves_unless_an_override_runs_it() {
    let planning = ServiceStream {
        service: device(0),
        stream: stream(9),
    };
    assert_eq!(LaunchMode::planning(planning), LaunchMode::Resolve);
    let measuring = STREAM_MODES.set(StreamMode::Execute, planning);
    assert_eq!(LaunchMode::planning(planning), LaunchMode::Execute);
    core::mem::drop(measuring);
    let gathering = STREAM_MODES.set(StreamMode::Discard, planning);
    assert_eq!(LaunchMode::planning(planning), LaunchMode::Resolve);
    core::mem::drop(gathering);
    assert!(LaunchMode::Resolve.discards_launch());
}

/// Under a process mode that discards every launch, a planning stream's launches follow that
/// mode: they compile in its batches and leave lazy pages unmapped.
#[test]
#[serial_test::serial]
fn a_planning_stream_follows_a_discarding_process_mode() {
    let collector = StatisticsCollector::new();
    let planning = ServiceStream {
        service: device(0),
        stream: stream(10),
    };
    let compile = ProcessModeOverride::new(ProcessMode::CompileOnly, &collector);
    assert_eq!(LaunchMode::planning(planning), LaunchMode::Queue);
    core::mem::drop(compile);
    let tune = ProcessModeOverride::new(ProcessMode::CompileAndAutotune, &collector);
    assert_eq!(LaunchMode::planning(planning), LaunchMode::Compile);
    core::mem::drop(tune);
}

/// A stream's mode overrides the process mode's default for that stream of
/// that device alone, the newest override deciding, and overrides drop
/// in any order.
#[test]
#[serial_test::serial]
fn a_stream_mode_holds_for_its_stream_on_its_device() {
    let collector = StatisticsCollector::new();
    let tune = ProcessModeOverride::new(ProcessMode::CompileAndAutotune, &collector);
    let measured = ServiceStream {
        service: device(0),
        stream: stream(1),
    };
    let other_stream = ServiceStream {
        stream: stream(2),
        ..measured
    };
    let other_device = ServiceStream {
        service: device(1),
        ..measured
    };

    let measuring = STREAM_MODES.set(StreamMode::Execute, measured);
    assert_eq!(LaunchMode::new(measured), LaunchMode::Execute);
    assert_eq!(LaunchMode::new(other_stream), LaunchMode::Compile);
    assert_eq!(LaunchMode::new(other_device), LaunchMode::Compile);

    let nested = STREAM_MODES.set(StreamMode::Discard, measured);
    assert_eq!(
        LaunchMode::new(measured),
        LaunchMode::Compile,
        "the newest decides"
    );
    core::mem::drop(measuring);
    assert_eq!(LaunchMode::new(measured), LaunchMode::Compile);
    core::mem::drop(nested);
    assert_eq!(
        LaunchMode::new(measured),
        LaunchMode::Compile,
        "the mode's again"
    );
    core::mem::drop(tune);
}
