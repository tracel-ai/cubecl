use core::num::NonZeroU64;

/// The id a tainted allocation carries, naming the failure that left its
/// bytes unwritten.
///
/// Opaque on purpose: a carrier gains a word and nothing else. It does not
/// learn what an error is, it cannot report one, and it has no opinion about
/// streams. `NonZero` so `Option<FailureId>` is that one word rather than two.
///
/// Wide because ids are never reused: one is minted for every write scope
/// that claims anything, which is every launch and every host copy, and a
/// narrower counter would run out. Reuse is what a 32-bit id would need, and
/// reuse is unsound here — [`prune`](super::ErrorGraph::prune) drops a node whose tag
/// count is zero, which a freshly minted node also has, so a recycled id lets
/// one scope's prune delete another's failure. Free in the carrier either way:
/// [`Tainted`](super::Taint) pads a `u32` out to eight bytes before its
/// ranges, which `a_failure_id_is_free_in_the_carrier` pins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FailureId(pub(super) NonZeroU64);

impl core::fmt::Display for FailureId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "#{}", self.0)
    }
}
