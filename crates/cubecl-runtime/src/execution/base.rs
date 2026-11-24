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
mod tests;
