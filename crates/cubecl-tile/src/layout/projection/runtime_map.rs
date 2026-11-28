//! The runtime side of a [`Projection`](crate::Projection): [`RuntimeMap`].

use cubecl::prelude::*;

use crate::Coords;

/// The runtime half of a [`Projection`](crate::Projection): its dynamic coefficients and divisors,
/// the window origin's phase under each divisor, where the tile starts along each dimension read
/// as digits, and the digits' place values and extents read off the buffer.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct RuntimeMap {
    /// Dynamic coefficients and divisors, physical axis major, each dimension's divisor last.
    pub(crate) coefficients: Coords<u32>,
    /// The window origin's phase within its divisor per physical axis, `0` when integer.
    pub(crate) residues: Coords<u32>,
    /// Where the tile starts along each dimension the projection reads digits of, in
    /// [`digit_dimensions`](crate::Projection::digit_dimensions) order: a window is never moved
    /// along a digit, so a read adds this before taking it.
    pub(crate) digit_starts: Coords<u32>,
    /// The [`Radix::Dynamic`](crate::Radix) place values and extents of the digits, in
    /// [`digit_radix_index`](crate::Projection::digit_radix_index) order.
    pub digit_radices: Coords<u32>,
}

#[cube]
impl RuntimeMap {
    /// The runtime map of an integer mapping: no coefficients, zero phases, no folded dimension.
    pub(crate) fn integral(#[comptime] physical_rank: usize) -> RuntimeMap {
        RuntimeMap {
            coefficients: Coords::<u32>::new(),
            residues: Coords::constant(comptime!(vec![0; physical_rank])),
            digit_starts: Coords::<u32>::new(),
            digit_radices: Coords::<u32>::new(),
        }
    }

    /// This map copied into mutable registers that survive a sibling slot's refill.
    pub(crate) fn stored(&self) -> RuntimeMap {
        RuntimeMap {
            coefficients: self.coefficients.stored(),
            residues: self.residues.stored(),
            digit_starts: self.digit_starts.stored(),
            digit_radices: self.digit_radices.stored(),
        }
    }

    /// Store a source window's complete runtime addressing state in this slot.
    pub(crate) fn store_from(&mut self, src: &RuntimeMap) {
        self.coefficients.store_from(&src.coefficients);
        self.residues.store_from(&src.residues);
        self.digit_starts.store_from(&src.digit_starts);
        self.digit_radices.store_from(&src.digit_radices);
    }
}
