use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::{MatrixLayout, stage::StageMemoryConfig};

#[derive(CubeType, Clone, Copy)]
/// Tile with a linear major dimension, and a strided minor dimension.
/// Basic tile kind supported by all stage matmuls.
pub struct StridedTile<ES: Numeric, IO: SliceVisibility = ReadOnly> {
    /// Slice containing all data
    pub slice: Slice<Line<ES>, IO>,
    /// Stride between each row/col, depending on MatrixLayout (the other is assumed to be 1)
    pub stride: u32,
    #[cube(comptime)]
    /// Layout of the tile (row-major or column-major).
    pub layout: MatrixLayout,
}

#[cube]
impl<ES: Numeric> StridedTile<ES> {
    /// Creates a tile from a contiguous slice of data.
    ///
    /// The slice length must exactly match the tile size.
    pub fn new_contiguous(
        slice: Slice<Line<ES>>,
        #[comptime] config: StageMemoryConfig,
    ) -> StridedTile<ES> {
        let layout = config.matrix_layout;
        let stride = match layout {
            MatrixLayout::RowMajor => config.elements_in_tile_col,
            MatrixLayout::ColMajor => config.elements_in_tile_row,
        };

        let stride = comptime![stride / config.stage_line_size];

        StridedTile::<ES> {
            slice,
            stride,
            layout,
        }
    }

    /// Creates a tile from a contiguous slice of data.
    ///
    /// The slice length must exactly match the tile size.
    pub fn new_contiguous_mut(
        slice: Slice<Line<ES>, ReadWrite>,
        #[comptime] config: StageMemoryConfig,
    ) -> StridedTile<ES, ReadWrite> {
        let layout = config.matrix_layout;
        let stride = match layout {
            MatrixLayout::RowMajor => config.elements_in_tile_col,
            MatrixLayout::ColMajor => config.elements_in_tile_row,
        };

        let stride = comptime![stride / config.stage_line_size];

        StridedTile::<ES, ReadWrite> {
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
    ) -> StridedTile<ES> {
        StridedTile::<ES> {
            slice,
            stride,
            layout,
        }
    }

    /// Creates a tile from a strided slice of data.
    ///
    /// The slice must include all elements of the tile, though it may include unused gaps.
    pub fn new_strided_mut(
        slice: Slice<Line<ES>, ReadWrite>,
        stride: u32,
        #[comptime] layout: MatrixLayout,
    ) -> StridedTile<ES, ReadWrite> {
        StridedTile::<ES, ReadWrite> {
            slice,
            stride,
            layout,
        }
    }
}

#[cube]
impl<ES: Numeric, IO: SliceVisibility> StridedTile<ES, IO> {
    /// Returns the tile as an unlined (scalar) slice.
    ///
    /// Returns:
    /// - The unlined slice
    /// - The updated stride to account for line width removal
    pub fn as_unlined(&self) -> (Slice<ES, IO>, u32) {
        let stage_line_size = comptime![self.slice.line_size()];
        (
            self.slice.try_cast_unchecked(),
            self.stride * stage_line_size,
        )
    }

    /// Returns a specific line from the tile based on coordinates.
    pub fn get_line(&self, coor_strided: u32, coor_contiguous: u32) -> Line<ES> {
        self.slice[coor_strided * self.stride + coor_contiguous]
    }
}
