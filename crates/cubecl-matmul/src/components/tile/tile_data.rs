use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::{Ident, InputIdent, MatrixLayout, tile::TileConfig};

#[derive(CubeType, Clone)]
/// Data to be handed to the tile matmul
pub struct Tile<ES: Numeric> {
    /// Slice containing all data
    pub slice: Slice<Line<ES>>,
    /// Stride between each row/col, depending on MatrixLayout (the other is assumed to be 1)
    pub stride: u32,
    #[cube(comptime)]
    pub layout: MatrixLayout,
}

#[cube]
impl<ES: Numeric> Tile<ES> {
    pub fn new_contiguous<T: TileConfig>(
        slice: Slice<Line<ES>>,
        #[comptime] ident: Ident,
        #[comptime] config: T,
    ) -> Tile<ES> {
        let layout = config.matrix_layout(ident);
        let stride = comptime! {
            (match ident.as_input_ident() {
            InputIdent::Lhs => match layout {
                MatrixLayout::RowMajor => config.tile_size().k(),
                MatrixLayout::ColMajor => config.tile_size().m(),
            },
            InputIdent::Rhs => match layout {
                MatrixLayout::RowMajor => config.tile_size().n(),
                MatrixLayout::ColMajor => config.tile_size().k(),
            },
        }) / config.stage_line_size(ident)};

        Tile::<ES> {
            slice,
            stride,
            layout,
        }
    }

    pub fn new_strided(
        slice: Slice<Line<ES>>,
        stride: u32,
        #[comptime] layout: MatrixLayout,
    ) -> Tile<ES> {
        Tile::<ES> {
            slice,
            stride,
            layout,
        }
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

    pub fn get_line(&self, strided: u32, contiguous: u32) -> Line<ES> {
        self.slice[strided * self.stride + contiguous]
    }

    pub fn get_segment_as_slice(&self, index: u32, #[comptime] num_lines: u32) -> Slice<Line<ES>> {
        let start = index * self.stride;
        self.slice.slice(start, start + num_lines)
    }
}
