use crate::matmul::matrix_layout::MatrixLayout;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
pub struct RefTile<'a, E: Numeric> {
    pub x: u32,
    pub y: u32,
    pub slice: &'a Slice<'a, Line<E>>,
    pub layout: MatrixLayout,
}

#[derive(CubeType)]
pub struct OwnedTile<E: Numeric> {
    pub x: u32,
    pub y: u32,
    pub handle: Array<E>,
    pub layout: MatrixLayout,
}

#[cube]
pub fn new_ref_tile<'a, E: Numeric>(
    x: u32,
    y: u32,
    slice: &'a Slice<'a, Line<E>>,
    layout: MatrixLayout,
) -> RefTile<'a, E> {
    RefTile::<'a, E> {
        x,
        y,
        slice,
        layout,
    }
}
