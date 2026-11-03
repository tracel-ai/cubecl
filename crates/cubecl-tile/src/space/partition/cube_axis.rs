//! [`CubeAxis`]: one of the launch grid's three dimensions.

use cubecl::prelude::*;

/// One of the launch grid's three dimensions.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum CubeAxis {
    X,
    Y,
    Z,
}

#[cube]
impl CubeAxis {
    /// This cube's position on grid dimension `dim`.
    pub(crate) fn position(#[comptime] dim: CubeAxis) -> usize {
        let cube_pos = match comptime!(dim) {
            CubeAxis::X => CUBE_POS_X,
            CubeAxis::Y => CUBE_POS_Y,
            CubeAxis::Z => CUBE_POS_Z,
        };
        cube_pos as usize
    }
}
