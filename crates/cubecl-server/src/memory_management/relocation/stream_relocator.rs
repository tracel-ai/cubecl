use super::{RelocationReason, StreamRelocationState, StreamsSubmitted};
use crate::memory_management::ErrorGraph;

/// Relocates one stream's memory: what a runtime whose streams share one way
/// to finish their work (a scheduler, say) supplies per stream, leaving the
/// rest of [`DeviceRelocator`](super::DeviceRelocator) to that shared way.
///
/// # Assumptions
///
/// `failures` is the [`ErrorGraph`] that holds the failures the stream's
/// memory carries.
pub trait StreamRelocator {
    /// What a relocation reads of the stream's memory: its need, unresolved,
    /// and the bytes it holds from the device.
    ///
    /// # Guarantees
    ///
    /// The need is [`RelocationNeed::Recording`](super::RelocationNeed::Recording) whenever a stream of the
    /// device records a graph.
    fn relocation_state(&self) -> StreamRelocationState;

    /// Move what the stream's memory holds on outdated pages, with a copier
    /// whose wait covers the work `submitted` proves was handed to the device.
    fn relocate(
        &mut self,
        reason: RelocationReason,
        submitted: &StreamsSubmitted,
        failures: &mut ErrorGraph,
    );

    /// Give back every page the stream's memory holds and nothing needs.
    fn cleanup_memory(&mut self, failures: &mut ErrorGraph);
}
