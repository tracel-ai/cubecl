//! A unit's place in the team it shares a leaf with.

use cubecl::prelude::*;

/// A unit's index in its team and the team's unit count.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct TeamUnit {
    /// This unit, among the team's.
    pub index: usize,
    /// Units the team holds.
    pub units: usize,
}

#[cube]
impl TeamUnit {
    /// A team laid along the cube's x dim: one team per row of the cube.
    pub fn along_x() -> TeamUnit {
        TeamUnit {
            index: UNIT_POS_X as usize,
            units: CUBE_DIM_X as usize,
        }
    }
}
