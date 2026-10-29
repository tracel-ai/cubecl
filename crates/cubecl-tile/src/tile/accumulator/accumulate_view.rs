//! The view a register block accumulates through ([`AccumulateView`]), what it starts from
//! ([`InitFrom`]) and where it reads a cell's own value ([`CellRead`]).

use cubecl::prelude::*;

use cubecl_std::tensor::layout::{Coordinates, Coords2d};

use crate::*;

/// What an accumulation starts from.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum InitFrom {
    /// Combine with the cell's existing value.
    Cell,
    /// Start from the monoid's identity; the cell is never read.
    Identity,
}

/// Where the cell's own value enters the result: at most one site reads it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum CellRead {
    /// This unit owns the cell whole, so the block starts from it.
    AtSeed,
    /// The units each hold a partial, so the elected writer combines the cell in once.
    AtCommit,
    /// Nothing the cell holds counts.
    Never,
}

impl CellRead {
    /// Derived from `init_from` and the unit share. A summing destination is never read here:
    /// its store, atomic or relayed, is the sum.
    const fn of(unit_share: UnitShare, init_from: InitFrom, write: Write) -> Self {
        match write {
            Write::Accumulate | Write::Exclusive(_) => CellRead::Never,
            Write::Replace => match init_from {
                InitFrom::Identity => CellRead::Never,
                InitFrom::Cell => match unit_share {
                    UnitShare::Repeated | UnitShare::Whole => CellRead::AtSeed,
                    UnitShare::Plane | UnitShare::Group { .. } => CellRead::AtCommit,
                },
            },
        }
    }
}

/// The view a register block accumulates through: [`seed`](AccumulateView::seed) it, contract into
/// it, [`commit`](AccumulateView::commit) it back.
#[derive(CubeType)]
pub(crate) struct AccumulateView<'a, E: Numeric, V: Size, C: Coordinates + 'a = Coords2d> {
    values: MaskedMut<'a, Vector<E, V>, C>,
    #[cube(comptime)]
    units: UnitShare,
    #[cube(comptime)]
    monoid: Monoid,
    #[cube(comptime)]
    cell_read: CellRead,
    #[cube(comptime)]
    drain: Drain,
}

#[cube]
impl<'a, E: Numeric, V: Size, C: Coordinates + 'a> AccumulateView<'a, E, V, C> {
    pub(crate) fn new(
        values: MaskedMut<'a, Vector<E, V>, C>,
        #[comptime] units: UnitShare,
        #[comptime] split_share: SplitShare,
        #[comptime] write: Write,
        #[comptime] monoid: Monoid,
        #[comptime] init_from: InitFrom,
    ) -> Self {
        comptime!(write.admits(split_share, monoid, "AccumulateView"));
        AccumulateView::<'a, E, V, C> {
            values,
            units,
            monoid,
            cell_read: comptime!(CellRead::of(units, init_from, write)),
            drain: comptime!(Drain::of(units, write)),
        }
    }

    /// The underlying overhang-mask flag.
    pub(crate) fn check(&self) -> comptime_type!(bool) {
        comptime!(self.values.check)
    }

    /// How these cells are shared across the plane's units. Past `Whole`,
    /// [`commit`](Self::commit) is a plane op and must not run under divergent control flow.
    pub(crate) fn unit_share(&self) -> comptime_type!(UnitShare) {
        comptime!(self.units)
    }

    /// The monoid these cells combine under.
    pub(crate) fn monoid(&self) -> comptime_type!(Monoid) {
        comptime!(self.monoid)
    }

    /// Whether a non-empty output block is wholly valid for unchecked seed/commit accesses.
    pub(crate) fn block_in_bounds(&self, pos: C, extent: C) -> bool {
        self.values.block_in_bounds(pos, extent)
    }

    /// A block's starting value: the cell where this site reads it, else the monoid's identity.
    pub(crate) fn seed(&self, pos: C) -> Vector<E, V> {
        match comptime!(self.cell_read) {
            CellRead::AtSeed => self.values.read(pos),
            CellRead::AtCommit | CellRead::Never => {
                Vector::<E, V>::cast_from(Monoid::identity::<E>(self.monoid))
            }
        }
    }

    /// Reduce a finished block across the plane; one unit writes the total.
    pub(crate) fn commit(&mut self, pos: C, value: Vector<E, V>) {
        match comptime!(self.drain) {
            Drain::PlaneReduce => {
                let combined = self.units.reduce::<Vector<E, V>>(value, self.monoid);
                self.commit_shared(pos, combined, UNIT_POS_X == 0);
            }
            Drain::GroupReduce { unit_bits } => {
                let combined = self.units.reduce::<Vector<E, V>>(value, self.monoid);
                let unit_in_group = UNIT_POS_X & comptime!(unit_bits as u32);
                self.commit_shared(pos, combined, unit_in_group == 0);
            }
            // A combine from repeating units would land once per unit, so one makes it.
            Drain::UnitZero => self.commit_shared(pos, value, UNIT_POS_X == 0),
            Drain::EachUnit => self.values.write(pos, value),
        }
    }

    /// Commit a shared cell under the one unit elected to write it.
    fn commit_shared(&mut self, pos: C, combined: Vector<E, V>, leader: bool) {
        if leader {
            match comptime!(self.cell_read) {
                CellRead::AtCommit => {
                    let old = self.values.read(pos.clone());
                    self.values
                        .write(pos, self.monoid.combine::<Vector<E, V>>(old, combined));
                }
                CellRead::AtSeed | CellRead::Never => self.values.write(pos, combined),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A destination that adds, atomically or in turns, is never read back by the drain, and
    /// units repeating each other's cells elect one to add them, or the cell would take the value
    /// once per unit.
    #[test]
    fn a_destination_that_adds_is_written_once_and_never_read() {
        for write in [Write::Accumulate, Write::Exclusive(Schedule::Sequential)] {
            assert_eq!(Drain::of(UnitShare::Repeated, write), Drain::UnitZero);
            for init_from in [InitFrom::Cell, InitFrom::Identity] {
                assert_eq!(
                    CellRead::of(UnitShare::Whole, init_from, write),
                    CellRead::Never
                );
            }
        }
        assert_eq!(
            Drain::of(UnitShare::Repeated, Write::Replace),
            Drain::EachUnit
        );
    }
}
