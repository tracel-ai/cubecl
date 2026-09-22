//! A relocation as a server runs it: across every stream, since any of them
//! may use the memory that moves.

use super::{RelocationNeed, RelocationReason};
use crate::memory_management::ErrorGraph;
use crate::server::ServerError;

/// One stream, as relocating its memory needs it: what a runtime whose
/// streams share one way to finish their work (a scheduler, say) supplies per
/// stream, leaving the rest of [`RelocatingStreams`] to that shared way.
pub trait RelocatableStream {
    /// Whether the stream is recording a graph.
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

/// A runtime's streams, as relocating one stream's memory needs them.
///
/// A relocation moves memory any stream may still read or write, so every
/// stream's work is done before a byte moves; and nothing moves while a graph
/// records, since the pages it touches keep their addresses until it seals.
/// The runtime supplies each step. The order they run in lives here, once.
pub trait RelocatingStreams {
    /// Whether any stream is recording a graph.
    fn recording(&mut self) -> bool;

    /// Whether a growth left pages behind in the relocating stream's memory:
    /// without any, a relocation has nothing to move and waits on nothing.
    fn has_outdated(&mut self) -> bool;

    /// Whether the relocating stream's memory wants a relocation, before the
    /// bytes the device holds are known.
    fn relocation_need(&mut self) -> RelocationNeed;

    /// The bytes every stream's memory holds from the device.
    fn bytes_allocated(&mut self) -> u64;

    /// Get every stream's queued and submitted work done, or leave it to the
    /// [`CopyQueue`](super::CopyQueue) to wait for before its first copy.
    fn finish(&mut self);

    /// Move what the relocating stream's memory holds on outdated pages.
    fn relocate_memory(&mut self, reason: RelocationReason);

    /// Give back every page the relocating stream's memory holds and nothing
    /// needs.
    fn cleanup_memory(&mut self);

    /// Refuse while any stream records a graph: the pages a recording
    /// touches keep their addresses until it seals, and nothing may wait on
    /// the device under it.
    ///
    /// What [`reclaim`](Self::reclaim) checks first, for a caller with work of
    /// its own to refuse before it.
    ///
    /// # Errors
    ///
    /// [`ServerError`] naming the recording when a stream records.
    fn refuse_while_recording(&mut self) -> Result<(), ServerError> {
        if self.recording() {
            return Err(ServerError::graph_state(
                "memory_cleanup: a stream is recording a graph, whose pages keep their addresses \
                 until it seals",
            ));
        }
        Ok(())
    }

    /// Give back everything the relocating stream's memory holds and nothing
    /// needs: what the outdated pages still hold moves first, while the
    /// current pages keep the room it moves into, then every page left empty
    /// goes back.
    ///
    /// # Errors
    ///
    /// Refused while any stream records a graph, see
    /// [`refuse_while_recording`](Self::refuse_while_recording).
    fn reclaim(&mut self) -> Result<(), ServerError> {
        self.refuse_while_recording()?;
        self.relocate(RelocationReason::Explicit);
        self.cleanup_memory();
        Ok(())
    }

    /// Relocate when the memory asks for it: while the device still has a
    /// page to spare, or before the pools run out of page sizes.
    fn relocate_when_wanted(&mut self) {
        let need = self.relocation_need();
        if need == RelocationNeed::Nothing || self.recording() {
            return;
        }
        if let Some(reason) = need.reason(|| self.bytes_allocated()) {
            self.finish();
            self.relocate_memory(reason);
        }
    }

    /// Relocate for `reason`, unless a graph records or nothing is outdated.
    fn relocate(&mut self, reason: RelocationReason) {
        if self.recording() || !self.has_outdated() {
            return;
        }
        self.finish();
        self.relocate_memory(reason);
    }
}
