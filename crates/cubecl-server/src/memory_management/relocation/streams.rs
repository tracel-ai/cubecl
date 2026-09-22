//! A relocation as a server runs it: across every stream, since any of them
//! may use the memory that moves.

use super::{RelocationNeed, RelocationReason};

/// A runtime's streams, as relocating one stream's memory needs them.
///
/// A relocation moves memory any stream may still read or write, so every
/// stream's work is done before a byte moves; and nothing moves while a graph
/// records, since the pages it touches keep their addresses until it seals.
/// The runtime supplies each step. The order they run in lives here, once.
pub trait RelocatingStreams {
    /// Whether any stream is recording a graph.
    fn recording(&mut self) -> bool;

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

    /// Relocate for `reason`, unless a graph records.
    fn relocate(&mut self, reason: RelocationReason) {
        if self.recording() {
            return;
        }
        self.finish();
        self.relocate_memory(reason);
    }
}
