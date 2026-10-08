use super::policy::policy;
use super::stream_mode::stream_mode;
use super::{ExecutionPolicy, StreamMode};
use cubecl_common::device::ServiceId;
use cubecl_environment::stream::StreamId;

/// What a server does with one launch: the verdict its stream's mode and the
/// process's policy resolve to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaunchAction {
    /// Compile if needed, then run it. The normal case.
    Execute,
    /// Compile if needed, cache the artifact, and drop the launch.
    ///
    /// A server honoring this must still do everything a first launch does
    /// short of dispatching — expand, compile, validate, populate its caches —
    /// or the pass buys nothing.
    Compile,
    /// Queue the kernel to be compiled with others, and drop the launch.
    ///
    /// A server honoring this compiles the queue when it next loads a kernel
    /// for a launch, and only then: flushing or syncing compiles nothing, so
    /// a pass that only queues gathers everything it reaches into one batch.
    /// A kernel that fails to compile there reports it when it is launched.
    Queue,
}

impl LaunchAction {
    /// Whether the launch is dropped rather than run.
    pub fn drops_launch(self) -> bool {
        matches!(self, Self::Compile | Self::Queue)
    }
}

/// One stream of one device: what a [`StreamModeOverride`] sets the mode of.
/// Stream ids are the process's, not a device's, so a measurement on one
/// device leaves the same stream of every other device alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceStream {
    /// The device's service, as its clients reach it.
    pub service: ServiceId,
    /// The stream on it.
    pub stream: StreamId,
}

/// What a launch issued on `on` does now.
pub fn launch_action(on: DeviceStream) -> LaunchAction {
    let policy = policy();
    let mode = stream_mode(on, policy.stream_default());
    match (mode, policy) {
        (StreamMode::Execute, _) => LaunchAction::Execute,
        // A stream dropping its launches while the process executes queues
        // them: they compile with the next launch that loads a kernel.
        (StreamMode::Compile, ExecutionPolicy::CompileOnly | ExecutionPolicy::Execute) => {
            LaunchAction::Queue
        }
        (StreamMode::Compile, ExecutionPolicy::CompileAndAutotune) => LaunchAction::Compile,
    }
}

#[cfg(test)]
mod tests {
    use super::super::{ExecutionOverride, StatisticsCollector, StreamModeOverride};
    use super::*;
    use cubecl_common::device::DeviceId;
    // `serial_test`'s macro expands to `vec!`, which a `no_std` crate has to
    // bring in itself.
    use alloc::vec;

    /// Stream `stream` of device `device`.
    fn on(device: u16, stream: u64) -> DeviceStream {
        DeviceStream {
            service: ServiceId::of::<()>(DeviceId {
                type_id: 0,
                index_id: device,
            }),
            stream: StreamId { value: stream },
        }
    }

    /// The verdict is a table of a stream's mode and the policy.
    #[test]
    #[serial_test::serial]
    fn a_launch_follows_its_stream_and_the_policy() {
        let collector = StatisticsCollector::new();
        let action = || launch_action(on(0, 7));
        assert_eq!(action(), LaunchAction::Execute);
        {
            let _compile = ExecutionOverride::new(ExecutionPolicy::CompileOnly, &collector);
            assert_eq!(action(), LaunchAction::Queue);
        }
        {
            let _tune = ExecutionOverride::new(ExecutionPolicy::CompileAndAutotune, &collector);
            assert_eq!(action(), LaunchAction::Compile);
        }
        {
            // A stream dropping its launches while the process executes
            // queues them.
            let _queuing = StreamModeOverride::on(StreamMode::Compile, on(0, 7));
            assert_eq!(action(), LaunchAction::Queue);
        }
        let _execute = ExecutionOverride::new(ExecutionPolicy::Execute, &collector);
        assert_eq!(action(), LaunchAction::Execute);
    }

    /// A stream's mode overrides the policy's default for that stream of
    /// that device alone, the newest override deciding, and overrides drop
    /// in any order.
    #[test]
    #[serial_test::serial]
    fn a_stream_mode_holds_for_its_stream_on_its_device() {
        let collector = StatisticsCollector::new();
        let _tune = ExecutionOverride::new(ExecutionPolicy::CompileAndAutotune, &collector);

        let measuring = StreamModeOverride::on(StreamMode::Execute, on(0, 1));
        assert_eq!(launch_action(on(0, 1)), LaunchAction::Execute);
        assert_eq!(
            launch_action(on(0, 2)),
            LaunchAction::Compile,
            "another stream"
        );
        assert_eq!(
            launch_action(on(1, 1)),
            LaunchAction::Compile,
            "another device"
        );

        let nested = StreamModeOverride::on(StreamMode::Compile, on(0, 1));
        assert_eq!(
            launch_action(on(0, 1)),
            LaunchAction::Compile,
            "the newest decides"
        );
        drop(measuring);
        assert_eq!(launch_action(on(0, 1)), LaunchAction::Compile);
        drop(nested);
        assert_eq!(
            launch_action(on(0, 1)),
            LaunchAction::Compile,
            "the policy's again"
        );
    }
}
