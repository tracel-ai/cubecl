use super::{ExecutionPolicy, ServiceStream, StreamMode};

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
    /// What a launch issued on `stream` does now.
    pub(crate) fn new(stream: ServiceStream) -> Self {
        let policy = ExecutionPolicy::current();
        match (stream.mode(policy.stream_mode()), policy) {
            (StreamMode::Execute, _) => Self::Execute,
            // A stream discarding its launches while the process executes
            // queues them: they compile with the next launch that loads a
            // kernel.
            (StreamMode::Discard, ExecutionPolicy::CompileOnly | ExecutionPolicy::Execute) => {
                Self::Queue
            }
            (StreamMode::Discard, ExecutionPolicy::CompileAndAutotune) => Self::Compile,
        }
    }

    /// Whether the launch is dropped rather than run.
    pub fn drops_launch(self) -> bool {
        matches!(self, Self::Compile | Self::Queue)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{ExecutionOverride, StatisticsCollector, StreamModeOverride};
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

    /// The verdict is a table of a stream's mode and the policy.
    #[test]
    #[serial_test::serial]
    fn a_launch_follows_its_stream_and_the_policy() {
        let collector = StatisticsCollector::new();
        let seventh = ServiceStream {
            service: device(0),
            stream: stream(7),
        };
        let action = || LaunchAction::new(seventh);
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
            // A stream discarding its launches while the process executes
            // queues them.
            let _queuing = StreamModeOverride::of_stream(StreamMode::Discard, seventh);
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

        let measuring = StreamModeOverride::of_stream(StreamMode::Execute, measured);
        assert_eq!(LaunchAction::new(measured), LaunchAction::Execute);
        assert_eq!(LaunchAction::new(other_stream), LaunchAction::Compile);
        assert_eq!(LaunchAction::new(other_device), LaunchAction::Compile);

        let nested = StreamModeOverride::of_stream(StreamMode::Discard, measured);
        assert_eq!(
            LaunchAction::new(measured),
            LaunchAction::Compile,
            "the newest decides"
        );
        drop(measuring);
        assert_eq!(LaunchAction::new(measured), LaunchAction::Compile);
        drop(nested);
        assert_eq!(
            LaunchAction::new(measured),
            LaunchAction::Compile,
            "the policy's again"
        );
    }
}
