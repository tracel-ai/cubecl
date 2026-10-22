//! How an accumulator's tiles reach memory: the plan ([`DrainPlan`]), the pass at each leaf
//! ([`DrainPass`]) and which units carry the writes ([`Drain`]).

use crate::*;

/// How a plane-resident accumulator's tiles reach memory, per its [`Scratch`] and its destination.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum DrainPlan {
    /// No scratch: each tile stores through its own intrinsic.
    Straight,
    /// Own slot per tile: all spills, then all adds, two barriers in all.
    BounceTogether,
    /// One shared slot: spill and add per tile, three barriers each.
    BounceEach,
}

impl DrainPlan {
    /// The plan for a grid holding `scratch`, into a destination its fragments must bounce to
    /// reach where `bounces`. A destination they store to whole takes them straight whatever the
    /// grid holds: the scratch is for the destinations that need it, and a grid draining into
    /// several pays its barriers only where one does.
    pub(crate) fn new(scratch: Scratch, bounces: bool) -> Self {
        match (scratch, bounces) {
            (Scratch::None, _) | (_, false) => DrainPlan::Straight,
            (Scratch::OneTile, true) => DrainPlan::BounceEach,
            (Scratch::WholeGrid, true) => DrainPlan::BounceTogether,
        }
    }
}

/// What a drain does to one tile at the leaf of its descent.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum DrainPass {
    /// Store through the tile's own intrinsic.
    Copy,
    /// Spill, add, and the three barriers around them.
    Bounce,
    /// Spill alone: the first pass.
    Spill,
    /// Add alone: the second pass.
    Add,
}

/// Which of the plane's units carry a drain's writes, and what they combine first.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Drain {
    /// Each unit holds whole cells of its own and writes them.
    EachUnit,
    /// Every unit holds the same whole cells and the write folds, so one unit writes.
    UnitZero,
    /// Units each hold a partial of one cell: combine across the plane, unit zero writes.
    PlaneFold,
    /// Groups each hold a partial of one cell: combine within the group, its first unit writes.
    GroupFold { unit_bits: usize },
}

impl Drain {
    pub(crate) const fn of(units: UnitShare, write: Write) -> Self {
        match (units, write) {
            (UnitShare::Plane, _) => Drain::PlaneFold,
            (UnitShare::Group { unit_bits }, _) => Drain::GroupFold { unit_bits },
            // Repeated units hold the same cells: a store may land many times, a fold only once.
            (UnitShare::Repeated, Write::Accumulate | Write::Exclusive(_)) => Drain::UnitZero,
            (UnitShare::Repeated, Write::Replace) | (UnitShare::Whole, _) => Drain::EachUnit,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A destination the fragments store to whole takes a grid straight, whatever scratch it
    /// holds: a grid opened with one for a sink pays no barrier on its way into a plain buffer.
    #[test]
    fn a_destination_stored_whole_takes_any_grid_straight() {
        for scratch in [Scratch::None, Scratch::OneTile, Scratch::WholeGrid] {
            assert_eq!(DrainPlan::new(scratch, false), DrainPlan::Straight);
        }
    }

    /// A destination the fragments bounce to reach takes them through the scratch the grid holds.
    #[test]
    fn a_bouncing_destination_takes_the_grid_through_its_scratch() {
        assert_eq!(
            DrainPlan::new(Scratch::OneTile, true),
            DrainPlan::BounceEach
        );
        assert_eq!(
            DrainPlan::new(Scratch::WholeGrid, true),
            DrainPlan::BounceTogether
        );
    }
}
