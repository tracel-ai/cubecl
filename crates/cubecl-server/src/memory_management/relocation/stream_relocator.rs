use super::{RelocationNeed, RelocationReason};
use crate::memory_management::ErrorGraph;

/// One stream, as relocating its memory needs it: what a runtime whose
/// streams share one way to finish their work (a scheduler, say) supplies per
/// stream, leaving the rest of [`DeviceRelocator`](super::DeviceRelocator) to that shared way.
pub trait StreamRelocator {
    /// Whether any stream of the device records a graph.
    fn recording(&self) -> bool;

    /// Whether a growth left pages behind in the stream's memory.
    fn has_outdated(&self) -> bool;

    /// Whether the stream's memory wants a relocation, see
    /// [`MemoryManagement::relocation_need`](crate::memory_management::MemoryManagement::relocation_need).
    fn relocation_need(&self) -> RelocationNeed;

    /// The bytes the stream's memory holds from the device.
    fn bytes_allocated(&self) -> u64;

    /// Move what the stream's memory holds on outdated pages, once every
    /// stream's work is done.
    fn relocate(&mut self, reason: RelocationReason, failures: &mut ErrorGraph);

    /// Give back every page the stream's memory holds and nothing needs.
    fn cleanup_memory(&mut self, failures: &mut ErrorGraph);
}
