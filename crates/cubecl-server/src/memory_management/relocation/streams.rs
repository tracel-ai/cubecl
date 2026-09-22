//! A relocation as a server runs it: across every stream, since any of them
//! may use the memory that moves.

use super::Relocate;

/// A runtime's streams, as relocating one stream's memory needs them.
///
/// A relocation moves memory any stream may still read or write, so every
/// stream's work is done before a byte moves; and nothing moves while a graph
/// records, since the pages it touches keep their addresses until it seals.
/// The runtime supplies each step. The order they run in lives here, once.
pub trait RelocatingStreams {
    /// Whether any stream is recording a graph.
    fn recording(&mut self) -> bool;

    /// Whether the relocating stream's memory has anything a relocation could
    /// move: the cheap question, asked before anything is summed.
    fn relocatable(&mut self) -> bool;

    /// The bytes every stream's memory holds from the device.
    fn bytes_allocated(&mut self) -> u64;

    /// Whether the relocating stream's memory asks for a relocation now, and
    /// why, on a device holding `allocated` bytes.
    fn relocation(&mut self, allocated: u64) -> Option<Relocate>;

    /// Get every stream's queued and submitted work done, or leave it to the
    /// [`CopyQueue`](super::CopyQueue) to wait for before its first copy.
    fn finish(&mut self);

    /// Move what the relocating stream's memory holds on outdated pages.
    fn relocate_memory(&mut self, reason: Relocate);

    /// Relocate when the memory asks for it: while the device still has a
    /// page to spare, or before the pools run out of page sizes.
    fn relocate_when_wanted(&mut self) {
        if !self.relocatable() || self.recording() {
            return;
        }
        let allocated = self.bytes_allocated();
        if let Some(reason) = self.relocation(allocated) {
            self.finish();
            self.relocate_memory(reason);
        }
    }

    /// Relocate for `reason`, unless a graph records.
    fn relocate(&mut self, reason: Relocate) {
        if self.recording() {
            return;
        }
        self.finish();
        self.relocate_memory(reason);
    }
}
