//! [`ComputeScope`]: the hardware a level's tiles can go to.

use cubecl::prelude::*;

/// The hardware a level's tiles can go to, ordered inner to outer.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum ComputeScope {
    Unit,
    Plane,
    /// `planes` consecutive planes, the first a multiple of `planes`, issuing one instruction
    /// together: a warpgroup MMA's four.
    PlaneGroup {
        planes: usize,
    },
    Cube,
}

impl ComputeScope {
    /// The four planes of a warpgroup, which issue each warpgroup MMA together
    /// ([`Instruction::Wgmma`](crate::Instruction::Wgmma)).
    pub const WARPGROUP: Self = Self::PlaneGroup { planes: 4 };

    /// The planes one instance of this scope spans, where it spans whole planes.
    pub fn planes(self) -> Option<usize> {
        match self {
            ComputeScope::Plane => Some(1),
            ComputeScope::PlaneGroup { planes } => Some(planes),
            ComputeScope::Unit | ComputeScope::Cube => None,
        }
    }
}

#[cube]
impl ComputeScope {
    /// This instance's position within `compute_scope`: which plane or plane group of the cube,
    /// or which unit of the plane.
    pub fn position(#[comptime] compute_scope: ComputeScope) -> usize {
        match comptime!(compute_scope) {
            ComputeScope::Plane => UNIT_POS_Y as usize,
            ComputeScope::PlaneGroup { planes } => UNIT_POS_Y as usize / planes,
            ComputeScope::Unit => UNIT_POS_X as usize,
            ComputeScope::Cube => {
                panic!(
                    "ComputeScope::position: a cube has one position per grid dimension; say which"
                )
            }
        }
    }

    /// This unit's position among the units of its instance of `compute_scope`: in the cube,
    /// in its plane group, or in its plane. A plane is the launch's `x`
    /// ([`Partitioning::cube_dim`](crate::Partitioning::cube_dim)), so a unit's position in it is
    /// `UNIT_POS_X`, and a group's planes are consecutive `y`s.
    pub fn unit(#[comptime] compute_scope: ComputeScope) -> usize {
        match comptime!(compute_scope) {
            ComputeScope::Cube => UNIT_POS as usize,
            ComputeScope::PlaneGroup { planes } => {
                (UNIT_POS_Y as usize % planes) * CUBE_DIM_X as usize + UNIT_POS_X as usize
            }
            ComputeScope::Plane => UNIT_POS_X as usize,
            ComputeScope::Unit => comptime!(panic!(
                "ComputeScope::unit: a unit is the only unit of its own scope"
            )),
        }
    }
}
