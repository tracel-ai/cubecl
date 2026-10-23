//! [`BoundaryPolicy`]: how the caller states an operand's bounds-check.

use crate::Boundary;

/// How an operand's reads are bounds-checked, where the caller states it. An operand that states
/// none is checked with [`Boundary::Zero`] on the axes that can leave the buffer.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BoundaryPolicy {
    /// Unchecked; the caller guarantees every read is in bounds.
    Unchecked,
    /// Checked with this boundary on every axis that is not provably in bounds.
    Every(Boundary),
    /// Checked with this boundary on the axes some extent could take it past, not only those this
    /// launch's do: every axis whose extent the kernel leaves dynamic and a tile can cut short,
    /// beside the static ones a tile overhangs. The kernel then checks the same axes whatever
    /// extents a launch brings, so the one compiled for one call serves every other — what a
    /// caller that compiles its kernels ahead of the calls needs, where a call's extents fall
    /// between the ones it compiled for.
    AnyExtent(Boundary),
}
