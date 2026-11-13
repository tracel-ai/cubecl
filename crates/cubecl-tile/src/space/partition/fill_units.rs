//! [`FillUnits`]: the units a level's tiles are taken in turns by, when they are not one each.

use cubecl::prelude::*;

use crate::{Axis, ComputeScope, Extent, Integer, IntegerExpand, Level, Region, Space, Walk};

/// The one axis of the lines a copy deals ([`FillUnits::deal`]).
const LINE: Axis = Axis(0);

/// The most turns a deal writes out one after another: past it, a loop strides by the units. A
/// runtime stride stalls each store on Metal's in-order pipe; a long run written out bloats the
/// kernel.
const MOST_WRITTEN_TURNS: usize = 8;

/// The units that take a level's tiles in turns: every unit of the cube, the units of the planes
/// the cube sets aside to fill its stages, of a plane group, of one plane, or one unit alone. What a copy's lines are
/// dealt to ([`Count::Dealt`](crate::Count)), and a plane's units taking a launch-sized level
/// ([`Levels::units_distributed`](crate::Levels::units_distributed)).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum FillUnits {
    /// Every unit of the cube, `units` of them where the launch states it.
    Cube { units: Option<usize> },
    /// The units of the `planes` planes at the end of the cube set aside to fill its stages.
    Fillers { planes: usize },
    /// The units of a group of `planes` consecutive planes.
    PlaneGroup { planes: usize },
    /// The units of one plane, `units` of them where the launch states it.
    Plane { units: Option<usize> },
    /// One unit, alone.
    Unit,
}

impl FillUnits {
    /// Every unit of the cube, `units` of them, where a binding states its launch's count
    /// ([`TileSpec::units`](crate::TileSpec), `0` where it states none).
    pub(crate) fn cube(units: usize) -> Self {
        FillUnits::Cube {
            units: (units > 0).then_some(units),
        }
    }

    /// The turns each of these units takes over `lines` lines, the last of them ragged, where
    /// their count is known.
    pub(crate) fn turns(self, lines: usize) -> Option<usize> {
        self.count().map(|units| lines.div_ceil(units))
    }

    /// How many there are, where the launch states it.
    pub(crate) fn count(self) -> Option<usize> {
        match self {
            FillUnits::Cube { units } | FillUnits::Plane { units } => units,
            FillUnits::Unit => Some(1),
            FillUnits::Fillers { .. } | FillUnits::PlaneGroup { .. } => None,
        }
    }

    /// The scope these units make up.
    pub(crate) fn scope(self) -> ComputeScope {
        match self {
            FillUnits::Cube { .. } | FillUnits::Fillers { .. } => ComputeScope::Cube,
            FillUnits::PlaneGroup { planes } => ComputeScope::PlaneGroup { planes },
            FillUnits::Plane { .. } => ComputeScope::Plane,
            FillUnits::Unit => ComputeScope::Unit,
        }
    }

    /// These units narrowed to `holder`'s where it has fewer: a plane's window of a buffer the
    /// cube fills is the plane's to fill.
    pub(crate) fn held_by(self, holder: ComputeScope) -> Self {
        match (holder < self.scope(), holder) {
            (false, _) => self,
            (true, ComputeScope::PlaneGroup { planes }) => FillUnits::PlaneGroup { planes },
            (true, ComputeScope::Plane) => FillUnits::Plane { units: None },
            (true, ComputeScope::Unit) => FillUnits::Unit,
            (true, ComputeScope::Cube) => unreachable!("nothing is wider than the cube"),
        }
    }
}

#[cube]
impl FillUnits {
    /// This unit's place among `fill`'s units: its place in the cube, among the filling planes'
    /// units, or in its plane. A plane is the launch's `x`, so a unit's place in it is
    /// `UNIT_POS_X`, and the filling planes are the cube's last.
    pub(crate) fn position(#[comptime] fill: FillUnits) -> usize {
        match fill {
            FillUnits::Cube { .. } => UNIT_POS as usize,
            FillUnits::Fillers { planes } => {
                (UNIT_POS - (CUBE_DIM - comptime!(planes as u32) * CUBE_DIM_X)) as usize
            }
            FillUnits::PlaneGroup { planes } => {
                ComputeScope::unit(comptime!(ComputeScope::PlaneGroup { planes }))
            }
            FillUnits::Plane { .. } => UNIT_POS_X as usize,
            FillUnits::Unit => 0usize.runtime(),
        }
    }

    /// How many units `fill` names: a constant where the launch states it, read off the launch
    /// otherwise.
    pub(crate) fn units(#[comptime] fill: FillUnits) -> usize {
        match comptime!(fill.count()) {
            Some(units) => units.runtime(),
            None => match fill {
                FillUnits::Cube { .. } => CUBE_DIM as usize,
                FillUnits::Fillers { planes } | FillUnits::PlaneGroup { planes } => {
                    planes * CUBE_DIM_X as usize
                }
                FillUnits::Plane { .. } => CUBE_DIM_X as usize,
                FillUnits::Unit => comptime!(unreachable!("one unit is counted")),
            },
        }
    }

    /// A copy's `lines` lines dealt to `fill`'s units in turns: unit `w` of `n` takes lines `w`,
    /// `w + n`, and on, so neighbouring units take neighbouring lines at once. Each region holds
    /// one line ([`line`](FillUnits::line)). The turns are written out where each unit takes a
    /// known few of them, the last guarded where it can run past the lines.
    pub(crate) fn deal(#[comptime] fill: FillUnits, lines: usize) -> Walk {
        let known = lines.constant();
        let extent = comptime!(match known {
            Some(lines) => Extent::Static(lines as usize),
            None => Extent::Dynamic,
        });
        let mut sizes = Sequence::<usize>::new();
        if comptime!(known.is_none()) {
            sizes.push(lines);
        }
        let space = Space::with_sizes(&comptime!(Space::from_extents(&[(LINE, extent)])), sizes);
        let walk =
            Region::rooted(&space, comptime!(vec![Level::dealt(LINE, 1, fill)]), 0usize).walk();
        let written = comptime!(
            known
                .and_then(|lines| fill.turns(lines as usize))
                .is_some_and(|turns| turns <= MOST_WRITTEN_TURNS)
        );
        if written { walk.unrolled() } else { walk }
    }

    /// The line a region of a [`deal`](FillUnits::deal) holds.
    pub(crate) fn line(region: &Region) -> usize {
        region.coord(LINE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_unit_takes_its_turns_and_the_last_may_be_ragged() {
        assert_eq!(FillUnits::cube(64).turns(256), Some(4));
        assert_eq!(FillUnits::cube(64).turns(200), Some(4));
        assert_eq!(FillUnits::cube(64).turns(1), Some(1));
        assert_eq!(FillUnits::cube(0).turns(256), None);
        assert_eq!(FillUnits::Fillers { planes: 1 }.turns(256), None);
    }

    #[test]
    fn a_window_held_by_fewer_units_is_theirs_to_fill() {
        let cube = FillUnits::cube(128);
        assert_eq!(cube.held_by(ComputeScope::Cube), cube);
        assert_eq!(
            cube.held_by(ComputeScope::Plane),
            FillUnits::Plane { units: None }
        );
        assert_eq!(cube.held_by(ComputeScope::Unit), FillUnits::Unit);
        let plane = FillUnits::Plane { units: Some(32) };
        assert_eq!(plane.held_by(ComputeScope::Cube), plane);
    }
}
