use super::{ProcessMode, STREAM_MODES, ServiceStream, StreamMode};

/// What a server does with one launch: the verdict its stream's mode and the
/// process mode resolve to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaunchMode {
    /// Compile if needed, then run it. The normal case.
    Execute,
    /// Compile if needed, cache the artifact, and discard the launch. A
    /// kernel of the work that fails to compile has no output to carry the
    /// failure, so the collector of the override open lists it
    /// ([`StatisticsCollector::work_failures`](super::StatisticsCollector::work_failures)).
    ///
    /// A server honoring this must still do everything a first launch does
    /// short of dispatching — expand, compile, validate, populate its caches —
    /// or the pass buys nothing.
    Compile,
    /// Queue the kernel to be compiled with others, and discard the launch.
    ///
    /// A server honoring this compiles the queue when it next loads a kernel
    /// for a launch, or when asked to (`Server::compile_queued`): flushing or
    /// syncing compiles nothing, so a pass that only queues gathers everything
    /// it reaches into one batch.
    /// A kernel that fails to compile there reports it when it is launched,
    /// and, when the work queued it, to the collector of the override it
    /// was queued under ([`StatisticsCollector::work_failures`](super::StatisticsCollector::work_failures)).
    Queue,
    /// Compile if needed, do everything the launch does on the host — resolve
    /// its buffers, build its metadata — and discard it before the dispatch:
    /// a launch of a graph capture's planning run. The recorded run then finds
    /// every buffer the launch path allocates already in the pools, since it
    /// may allocate none of its own. Nothing that reads the device runs, so a
    /// dynamic cube count is never resolved.
    Resolve,
}

impl LaunchMode {
    /// What a launch issued on `stream` does now.
    pub(crate) fn new(stream: ServiceStream) -> Self {
        let mode = ProcessMode::current();
        match (STREAM_MODES.mode(stream, mode.stream_mode()), mode) {
            (StreamMode::Execute, _) => Self::Execute,
            // A stream discarding its launches while the process executes
            // queues them: they compile with the next launch that loads a
            // kernel.
            (StreamMode::Discard, ProcessMode::CompileOnly | ProcessMode::Execute) => Self::Queue,
            (StreamMode::Discard, ProcessMode::CompileAndAutotune) => Self::Compile,
        }
    }

    /// What a launch issued on `stream` does while the stream plans a graph capture's work: it
    /// resolves, unless an override runs it (a tune measuring), since the recorded run must find
    /// everything the launch path allocates already in the pools.
    ///
    /// Under a process mode that discards every launch, it follows that mode instead: such a
    /// pass compiles in batches and maps pages lazily, and a resolving launch would compile on
    /// its own and give its buffers device backing. Its recording runs nothing either.
    pub(crate) fn planning(stream: ServiceStream) -> Self {
        let process = ProcessMode::current();
        match (STREAM_MODES.mode(stream, StreamMode::Discard), process) {
            (StreamMode::Execute, _) => Self::Execute,
            (StreamMode::Discard, ProcessMode::Execute) => Self::Resolve,
            (StreamMode::Discard, ProcessMode::CompileOnly) => Self::Queue,
            (StreamMode::Discard, ProcessMode::CompileAndAutotune) => Self::Compile,
        }
    }

    /// Whether the launch is discarded rather than run.
    pub fn discards_launch(self) -> bool {
        matches!(self, Self::Compile | Self::Queue | Self::Resolve)
    }
}

#[cfg(test)]
mod tests {
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
}
