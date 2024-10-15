use crate::matmul::matrix_layout::MatrixLayout;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
pub struct OwnedTile<E: Numeric> {
    pub x: u32,
    pub y: u32,
    pub handle: Array<E>,
    pub layout: MatrixLayout,
}
