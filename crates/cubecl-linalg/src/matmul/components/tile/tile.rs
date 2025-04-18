use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::{Ident, InputIdent, MatrixLayout, stage::Skew};

use super::TileConfig;

#[derive(CubeType)]
/// Data to be handed to the tile matmul
pub struct Tile<ES: Numeric> {
    /// Slice containing all data
    pub slice: Slice<Line<ES>>,
    /// Stride between each row/col, depending on MatrixLayout (the other is assumed to be 1)
    pub stride: u32,
}

#[cube]
impl<ES: Numeric> Tile<ES> {
    pub fn new_contiguous<T: TileConfig>(
        slice: Slice<Line<ES>>,
        #[comptime] skew: Skew,
        #[comptime] ident: Ident,
        #[comptime] config: T,
    ) -> Tile<ES> {
        comptime! {if let Skew::Pad(_) = skew {
            todo!()
        }}

        let stride = comptime! {
            (match ident.as_input_ident() {
            InputIdent::Lhs => match config.matrix_layout(ident) {
                MatrixLayout::RowMajor => config.tile_shape().k,
                MatrixLayout::ColMajor => config.tile_shape().m,
            },
            InputIdent::Rhs => match config.matrix_layout(ident) {
                MatrixLayout::RowMajor => config.tile_shape().n,
                MatrixLayout::ColMajor => config.tile_shape().k,
            },
        }) / config.stage_line_size(ident)};

        Tile::<ES> { slice, stride }
    }

    /// A tile whose segments are all within `slice` but may be spaced
    /// The stride should account for the skew
    pub fn new_strided(slice: Slice<Line<ES>>, stride: u32) -> Tile<ES> {
        Tile::<ES> { slice, stride }
    }

    pub fn as_unlined<T: TileConfig>(
        &self,
        #[comptime] ident: Ident,
        #[comptime] config: T,
    ) -> (Slice<ES>, u32) {
        (
            self.slice.try_cast_unchecked(),
            self.stride * config.stage_line_size(ident),
        )
    }
}
