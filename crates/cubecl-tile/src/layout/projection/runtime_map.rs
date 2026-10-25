//! The runtime side of a [`Projection`](crate::Projection): [`RuntimeMap`].

use cubecl::prelude::*;

use crate::Coords;

/// The runtime half of a [`Projection`](crate::Projection): its dynamic coefficients and divisors, and the
/// window origin's phase under each divisor.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct RuntimeMap {
    /// Dynamic coefficients and divisors, physical axis major, each axis's divisor last.
    pub(crate) coefficients: Coords<u32>,
    /// The window origin's phase within its divisor per physical axis, `0` when integer.
    pub(crate) residues: Coords<u32>,
}

#[cube]
impl RuntimeMap {
    /// The runtime map of an integer mapping: no coefficients, zero phases.
    pub(crate) fn integral(#[comptime] physical_rank: usize) -> RuntimeMap {
        RuntimeMap {
            coefficients: Coords::<u32>::new(),
            residues: Coords::constant(comptime!(vec![0; physical_rank])),
        }
    }

    /// This map copied into mutable registers that survive a sibling slot's refill.
    pub(crate) fn stored(&self) -> RuntimeMap {
        RuntimeMap {
            coefficients: self.coefficients.stored(),
            residues: self.residues.stored(),
        }
    }

    /// Store a source window's complete runtime addressing state in this slot.
    pub(crate) fn store_from(&mut self, src: &RuntimeMap) {
        self.coefficients.store_from(&src.coefficients);
        self.residues.store_from(&src.residues);
    }
}
