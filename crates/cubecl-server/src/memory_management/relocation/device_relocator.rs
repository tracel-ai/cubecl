//! A relocation as a server runs it: across every stream, since any of them
//! may use the memory that moves.

use super::{RelocationNeed, RelocationReason, RelocationScope};
use crate::server::ServerError;

/// Runs a relocation of one stream's memory, or every stream's, in the order
/// a relocation needs: every stream's work is done before a byte moves, since
/// a relocation moves memory any stream may still read or write; and nothing
/// moves while a graph records, since the pages it touches keep their
/// addresses until it seals.
///
/// A runtime supplies each step. The order they run in lives here, once.
pub trait DeviceRelocator {
    /// Whether the memory of `scope` wants a relocation, resolved against the
    /// bytes every stream's memory holds from the device.
    ///
    /// # Guarantees
    ///
    /// [`RelocationNeed::Recording`] whenever a stream records a graph,
    /// whatever the memory holds.
    fn relocation_need(&mut self, scope: RelocationScope) -> RelocationNeed;

    /// Get every stream's queued and submitted work done, or leave it to the
    /// [`RelocationCopier`](super::RelocationCopier) to wait for before its
    /// first copy.
    fn finish(&mut self);

    /// Move what the memory of `scope` holds on outdated pages, for `reason`.
    fn relocate_memory(&mut self, scope: RelocationScope, reason: RelocationReason);

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
        match self.relocation_need(RelocationScope::Stream) {
            RelocationNeed::Recording => Err(ServerError::graph_state(
                "memory_cleanup: a stream is recording a graph, whose pages keep their addresses \
                 until it seals",
            )),
            _ => Ok(()),
        }
    }

    /// Give back everything the relocating stream's memory holds and nothing
    /// needs: what the outdated pages still hold moves first, while the
    /// current pages keep the room it moves into, then every page left empty
    /// goes back.
    ///
    /// # Guarantees
    ///
    /// Moves nothing while a stream records a graph.
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

    /// Relocate the relocating stream's memory when it asks for it: while
    /// the device still has a page to spare, or before the pools run out of
    /// page sizes.
    ///
    /// # Guarantees
    ///
    /// Moves nothing while a stream records a graph.
    fn relocate_when_wanted(&mut self) {
        if let RelocationNeed::Wanted(reason) = self.relocation_need(RelocationScope::Stream) {
            self.finish_and_relocate(RelocationScope::Stream, reason);
        }
    }

    /// Relocate the relocating stream's memory for `reason`, unless a graph
    /// records or nothing is outdated.
    ///
    /// # Guarantees
    ///
    /// Moves nothing while a stream records a graph.
    fn relocate(&mut self, reason: RelocationReason) {
        self.relocate_outdated(RelocationScope::Stream, reason);
    }

    /// Relocate every stream's memory for `reason`, unless a graph records or
    /// nothing is outdated anywhere.
    ///
    /// What a capture does before its warmup: a recording may touch memory
    /// any stream owns, and every page it touches is guarded for the graph's
    /// life, where no relocation can empty it.
    ///
    /// # Guarantees
    ///
    /// Moves nothing while a stream records a graph.
    fn relocate_device(&mut self, reason: RelocationReason) {
        self.relocate_outdated(RelocationScope::Device, reason);
    }

    /// Relocate the memory of `scope` for `reason`, unless a graph records or
    /// nothing is outdated there.
    fn relocate_outdated(&mut self, scope: RelocationScope, reason: RelocationReason) {
        if self.relocation_need(scope).has_outdated() {
            self.finish_and_relocate(scope, reason);
        }
    }

    /// Get every stream's work done, then move what the memory of `scope`
    /// holds on outdated pages.
    fn finish_and_relocate(&mut self, scope: RelocationScope, reason: RelocationReason) {
        self.finish();
        self.relocate_memory(scope, reason);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// A device whose memory needs `need`, recording each step it is asked
    /// for.
    struct Recorded {
        need: RelocationNeed,
        steps: Vec<&'static str>,
    }

    impl DeviceRelocator for Recorded {
        fn relocation_need(&mut self, _scope: RelocationScope) -> RelocationNeed {
            self.need
        }

        fn finish(&mut self) {
            self.steps.push("finish");
        }

        fn relocate_memory(&mut self, _scope: RelocationScope, _reason: RelocationReason) {
            self.steps.push("relocate");
        }

        fn cleanup_memory(&mut self) {
            self.steps.push("cleanup");
        }
    }

    fn device(need: RelocationNeed) -> Recorded {
        Recorded {
            need,
            steps: Vec::new(),
        }
    }

    #[test]
    fn a_reclaim_is_refused_while_a_stream_records() {
        let mut device = device(RelocationNeed::Recording);

        assert!(device.reclaim().is_err());
        assert!(device.steps.is_empty());
    }

    #[test]
    fn a_reclaim_moves_what_is_outdated_then_cleans_up() {
        let mut device = device(RelocationNeed::Settled);

        device.reclaim().expect("nothing should record");

        assert_eq!(device.steps, ["finish", "relocate", "cleanup"]);
    }

    #[test]
    fn a_reclaim_with_nothing_outdated_only_cleans_up() {
        let mut device = device(RelocationNeed::Nothing);

        device.reclaim().expect("nothing should record");

        assert_eq!(device.steps, ["cleanup"]);
    }

    #[test]
    fn a_settled_relocation_does_not_run_unasked() {
        let mut settled = device(RelocationNeed::Settled);

        settled.relocate_when_wanted();

        assert!(settled.steps.is_empty());
    }

    #[test]
    fn a_wanted_relocation_runs_unasked() {
        let mut wanted = device(RelocationNeed::Wanted(RelocationReason::ArenaFull));

        wanted.relocate_when_wanted();

        assert_eq!(wanted.steps, ["finish", "relocate"]);
    }

    #[test]
    fn a_device_relocation_waits_for_a_recording() {
        let mut recording = device(RelocationNeed::Recording);

        recording.relocate_device(RelocationReason::Capture);

        assert!(recording.steps.is_empty());
    }

    #[test]
    fn a_device_relocation_runs_with_something_outdated() {
        let mut settled = device(RelocationNeed::Settled);

        settled.relocate_device(RelocationReason::Capture);

        assert_eq!(settled.steps, ["finish", "relocate"]);
    }
}
