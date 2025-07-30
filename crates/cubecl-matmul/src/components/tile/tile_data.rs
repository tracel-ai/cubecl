use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::{MatrixLayout, StageIdent, tile::TileConfig};

#[derive(CubeType, Clone)]
/// Data to be handed to the Tile Matmul
pub struct Tile<ES: Numeric> {
    /// Slice containing all data
    pub slice: Slice<Line<ES>>,
    /// Stride between each row/col, depending on MatrixLayout (the other is assumed to be 1)
    pub stride: u32,
    #[cube(comptime)]
    /// Layout of the tile (row-major or column-major).
    pub layout: MatrixLayout,
}

#[cube]
impl<ES: Numeric> Tile<ES> {
    /// Creates a tile from a contiguous slice of data.
    ///
    /// The slice length must exactly match the tile size.
    pub fn new_contiguous<T: TileConfig>(
        slice: Slice<Line<ES>>,
        #[comptime] ident: StageIdent,
        #[comptime] config: T,
    ) -> Tile<ES> {
        let layout = config.matrix_layout(ident);
        let stride = comptime! {
            (match ident {
            StageIdent::Lhs => match layout {
                MatrixLayout::RowMajor => config.tile_size().k(),
                MatrixLayout::ColMajor => config.tile_size().m(),
            },
            StageIdent::Rhs => match layout {
                MatrixLayout::RowMajor => config.tile_size().n(),
                MatrixLayout::ColMajor => config.tile_size().k(),
            },
            StageIdent::Acc => unreachable!()
        }) / config.stage_line_size(ident)};

        Tile::<ES> {
            slice,
            stride,
            layout,
        }
    }

    /// Creates a tile from a strided slice of data.
    ///
    /// The slice must include all elements of the tile, though it may include unused gaps.
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

    /// Returns the tile as an unlined (scalar) slice.
    ///
    /// Returns:
    /// - The unlined slice
    /// - The updated stride to account for line width removal
    pub fn as_unlined<T: TileConfig>(
        &self,
        #[comptime] ident: StageIdent,
        #[comptime] config: T,
    ) -> (Slice<ES>, u32) {
        (
            self.slice.try_cast_unchecked(),
            self.stride * config.stage_line_size(ident),
        )
    }

    /// Returns a specific line from the tile based on coordinates.
    pub fn get_line(&self, coor_strided: u32, coor_contiguous: u32) -> Line<ES> {
        self.slice[coor_strided * self.stride + coor_contiguous]
    }
}
