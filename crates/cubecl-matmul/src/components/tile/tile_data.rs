use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, intrinsic};
use cubecl_std::{Swizzle, type_size};

use crate::components::{
    MatrixLayout,
    stage::{StageMemoryConfig, as_swizzle_object},
};

#[derive(CubeType, Clone, Copy)]
/// Tile with a linear major dimension, and a strided minor dimension.
/// Basic tile kind supported by all stage matmuls.
pub struct StridedTile<ES: Numeric, IO: SliceVisibility = ReadOnly> {
    /// Slice containing all data for the stage
    pub stage: Slice<Line<ES>, IO>,
    /// Offset of the tile in the stage
    pub start: u32,
    /// End of the tile in the stage, may be wrong with swizzle
    pub end: u32,
    /// Stride between each row/col, depending on MatrixLayout (the other is assumed to be 1)
    pub stride: u32,
    /// Swizzle object to transform the index
    pub swizzle: Swizzle,
    #[cube(comptime)]
    /// Layout of the tile (row-major or column-major).
    pub layout: MatrixLayout,
    #[cube(comptime)]
    /// Line size of the slice
    pub line_size: u32,
}

#[cube]
impl<ES: Numeric> StridedTile<ES> {
    /// Creates a tile from a contiguous slice of data.
    ///
    /// The slice length must exactly match the tile size.
    pub fn new_contiguous(
        stage: Slice<Line<ES>>,
        start: u32,
        #[comptime] config: StageMemoryConfig,
    ) -> StridedTile<ES> {
        let len = config.elements_per_tile() / config.line_size;
        let layout = config.matrix_layout;
        let stride = match layout {
            MatrixLayout::RowMajor => config.elements_per_tile_along_col,
            MatrixLayout::ColMajor => config.elements_per_tile_along_row,
        };

        let stride = comptime![stride / config.line_size];

        StridedTile::<ES> {
            stage,
            start,
            end: start + len,
            stride,
            swizzle: as_swizzle_object(config.swizzle),
            layout,
            line_size: config.line_size,
        }
    }

    /// Creates a tile from a contiguous slice of data.
    ///
    /// The slice length must exactly match the tile size.
    pub fn new_contiguous_mut(
        stage: Slice<Line<ES>, ReadWrite>,
        start: u32,
        #[comptime] config: StageMemoryConfig,
    ) -> StridedTile<ES, ReadWrite> {
        let len = config.elements_per_tile() / config.line_size;
        let layout = config.matrix_layout;
        let stride = match layout {
            MatrixLayout::RowMajor => config.elements_per_tile_along_col,
            MatrixLayout::ColMajor => config.elements_per_tile_along_row,
        };

        let stride = comptime![stride / config.line_size];

        StridedTile::<ES, ReadWrite> {
            stage,
            start,
            end: start + len,
            stride,
            swizzle: as_swizzle_object(config.swizzle),
            layout,
            line_size: config.line_size,
        }
    }

    /// Creates a tile from a strided slice of data.
    ///
    /// The slice must include all elements of the tile, though it may include unused gaps.
    pub fn new_strided(
        stage: Slice<Line<ES>>,
        start: u32,
        end: u32,
        stride: u32,
        swizzle: Swizzle,
        #[comptime] layout: MatrixLayout,
        #[comptime] line_size: u32,
    ) -> StridedTile<ES> {
        StridedTile::<ES> {
            stage,
            start,
            end,
            stride,
            swizzle,
            layout,
            line_size,
        }
    }

    /// Creates a tile from a strided slice of data.
    ///
    /// The slice must include all elements of the tile, though it may include unused gaps.
    pub fn new_strided_mut(
        stage: Slice<Line<ES>, ReadWrite>,
        start: u32,
        end: u32,
        stride: u32,
        swizzle: Swizzle,
        #[comptime] layout: MatrixLayout,
        #[comptime] line_size: u32,
    ) -> StridedTile<ES, ReadWrite> {
        StridedTile::<ES, ReadWrite> {
            stage,
            start,
            end,
            stride,
            swizzle,
            layout,
            line_size,
        }
    }
}

#[cube]
impl<ES: Numeric> StridedTile<ES, ReadOnly> {
    /// Returns the tile as an unlined (scalar) slice.
    ///
    /// Returns:
    /// - The unlined slice
    /// - The updated stride to account for line width removal
    pub fn as_unlined(&self) -> (Slice<ES, ReadOnly>, u32) {
        let stage_line_size = comptime![self.stage.line_size()];
        (
            self.stage.slice(self.start, self.end).try_cast_unchecked(),
            self.stride * stage_line_size,
        )
    }
}

#[cube]
impl<ES: Numeric> StridedTile<ES, ReadWrite> {
    /// Returns the tile as an unlined (scalar) slice.
    ///
    /// Returns:
    /// - The unlined slice
    /// - The updated stride to account for line width removal
    pub fn as_unlined_mut(&self) -> (Slice<ES, ReadWrite>, u32) {
        let stage_line_size = comptime![self.stage.line_size()];
        (
            self.stage
                .slice(self.start, self.end)
                .as_mut_unchecked()
                .try_cast_unchecked(),
            self.stride * stage_line_size,
        )
    }

    /// Returns the tile as an offset slice. Should only be used when swizzling is definitely not
    /// applicable.
    pub fn as_slice_mut(&self) -> Slice<Line<ES>, ReadWrite> {
        self.stage.slice(self.start, self.end).as_mut_unchecked()
    }
}

#[cube]
impl<ES: Numeric, IO: SliceVisibility> StridedTile<ES, IO> {
    /// Returns a specific line from the tile based on coordinates.
    pub fn get_line(&self, coor_strided: u32, coor_contiguous: u32) -> Line<ES> {
        let offset = coor_strided * self.stride + coor_contiguous;
        let offset_abs = self.start + offset;
        let type_size = type_size::<ES>(self.stage.line_size());
        let offset_swizzled = self.swizzle.apply(offset_abs, type_size);
        self.stage[offset_swizzled]
    }

    pub fn stage_offset(&self, relative_offset: u32) -> u32 {
        let offset = self.start + relative_offset;
        let type_size = type_size::<ES>(self.stage.line_size());
        self.swizzle.apply(offset, type_size)
    }

    #[allow(unused_variables)]
    pub fn with_line_size(&self, #[comptime] line_size: u32) -> Self {
        intrinsic!(|scope| {
            let stage_line_size = self.stage.line_size();

            if line_size == self.stage.line_size() {
                return self;
            }

            let current = stage_line_size;
            let mut out = self.clone();

            if current < line_size {
                let ratio = line_size / current;
                let end = cubecl::frontend::div::expand(scope, self.end, ratio.into());
                let start = cubecl::frontend::div::expand(scope, self.start, ratio.into());
                let stride = cubecl::frontend::div::expand(scope, self.stride, ratio.into());
                out.start = start;
                out.end = end;
                out.stride = stride;
            } else {
                let ratio = current / line_size;
                let start = cubecl::frontend::mul::expand(scope, self.start, ratio.into());
                let end = cubecl::frontend::mul::expand(scope, self.end, ratio.into());
                let stride = cubecl::frontend::mul::expand(scope, self.stride, ratio.into());
                out.start = start;
                out.end = end;
                out.stride = stride;
            }

            out.stage = out.stage.__expand_with_line_size_method(scope, line_size);
            out
        })
    }
}
