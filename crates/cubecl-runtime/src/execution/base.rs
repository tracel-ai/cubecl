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
    /// a launch of a graph capture's rehearsal. The recorded run then finds
    /// every buffer the launch path allocates already in the pools, since it
    /// may allocate none of its own. Nothing that reads the device runs, so a
    /// dynamic cube count is never resolved.
    Rehearse,
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

    /// The mode of this launch issued on a stream that is capturing a graph: a
    /// launch discarded there belongs to the capture's rehearsal, and does the
    /// launch's host side so the recorded run allocates nothing the rehearsal
    /// did not.
    pub(crate) fn within_capture(self) -> Self {
        match self.discards_launch() {
            true => Self::Rehearse,
            false => self,
        }
    }

    /// Whether the launch is discarded rather than run.
    pub fn discards_launch(self) -> bool {
        matches!(self, Self::Compile | Self::Queue | Self::Rehearse)
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

    /// A launch a capturing stream discards belongs to the capture's rehearsal, so it
    /// rehearses; one it runs is recorded or measured, and stays as it is.
    #[test]
    fn a_discarded_launch_within_a_capture_rehearses() {
        assert_eq!(LaunchMode::Queue.within_capture(), LaunchMode::Rehearse);
        assert_eq!(LaunchMode::Compile.within_capture(), LaunchMode::Rehearse);
        assert_eq!(LaunchMode::Execute.within_capture(), LaunchMode::Execute);
        assert!(LaunchMode::Rehearse.discards_launch());
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
