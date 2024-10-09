use crate::matmul::matrix_layout::MatrixLayout;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
pub struct Tile<'a, E: Numeric> {
    pub x: u32,
    pub y: u32,
    pub slice: &'a Slice<'a, Line<E>>,
    pub layout: MatrixLayout,
}

#[cube]
pub fn new_tile<'a, E: Numeric>(
    x: u32,
    y: u32,
    slice: &'a Slice<'a, Line<E>>,
    layout: MatrixLayout,
) -> Tile<'a, E> {
    Tile::<'a, E> {
        x,
        y,
        slice,
        layout,
    }
}
