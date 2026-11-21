//! [`CubeAxis`]: one of the launch grid's three axes.

use cubecl::prelude::*;

/// One of the launch grid's three axes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum CubeAxis {
    X,
    Y,
    Z,
}

#[cube]
impl CubeAxis {
    /// This cube's position on grid axis `cube_axis`.
    pub(crate) fn position(#[comptime] cube_axis: CubeAxis) -> usize {
        let cube_pos = match cube_axis {
            CubeAxis::X => CUBE_POS_X,
            CubeAxis::Y => CUBE_POS_Y,
            CubeAxis::Z => CUBE_POS_Z,
        };
        cube_pos as usize
    }
}
