//! [`BoundaryPolicy`]: how the caller states an operand's bounds-check.

use crate::Boundary;

/// How an operand's reads are bounds-checked, where the caller states it. An operand that states
/// none is checked with [`Boundary::Zero`] on the axes that can leave the buffer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BoundaryPolicy {
    /// Unchecked; the caller guarantees every read is in bounds.
    Unchecked,
    /// Checked with this boundary on every axis that is not provably in bounds.
    Every(Boundary),
}
