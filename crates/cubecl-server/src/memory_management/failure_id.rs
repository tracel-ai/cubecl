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
/// a [`Taint`](super::Taint) claim pads a `u32` out to eight bytes before its
/// ranges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FailureId(NonZeroU64);

impl FailureId {
    /// The number the id is shown and reported as.
    pub fn to_u64(self) -> u64 {
        self.0.get()
    }
}

impl FailureId {
    /// The id the `minted`-th failure carries.
    pub(crate) fn from_minted(minted: NonZeroU64) -> Self {
        Self(minted)
    }
}

impl core::fmt::Display for FailureId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "#{}", self.0)
    }
}
