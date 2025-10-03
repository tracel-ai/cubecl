use cubecl::prelude::*;
use cubecl_core::{self as cubecl};
use cubecl_std::tensor::{
    layout::{Coords1d, Coords2d, Coords3d, Layout, LayoutExpand},
    r#virtual::VirtualTensor,
};

use crate::components::{MatrixLayout, global::memory::GlobalMemoryConfig};

/// Global layout that uses the last two dimensions and ignores all others.
#[derive(CubeType, Clone, Copy)]
pub struct SimpleGlobalLayout {
    rows: u32,
    stride_row: u32,
    columns: u32,
    stride_col: u32,
    batch_offset: u32,
    #[cube(comptime)]
    config: GlobalMemoryConfig,
}

#[cube]
impl SimpleGlobalLayout {
    /// Creates a new 2D layout starting at `batch_offset`.
    pub fn new<T: Numeric, IO: Clone>(
        tensor: &VirtualTensor<T, IO>,
        batch_offset: u32,
        #[comptime] config: GlobalMemoryConfig,
    ) -> Self {
        let rank = tensor.rank();

        SimpleGlobalLayout {
            rows: tensor.shape(rank - 2),
            stride_row: tensor.stride(rank - 2),
            columns: tensor.shape(rank - 1),
            stride_col: tensor.stride(rank - 1),
            batch_offset,
            config,
        }
    }
}

#[cube]
impl Layout for SimpleGlobalLayout {
    type Coordinates = Coords2d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, coords: Self::Coordinates) -> u32 {
        let line_size = comptime![self.config.global_line_size];
        let (row, col) = coords;
        let idx = self.batch_offset + row * self.stride_row + col * self.stride_col;

        idx / line_size
    }

    fn to_source_pos_checked(&self, coords: Self::Coordinates) -> (u32, bool) {
        (self.to_source_pos(coords), self.is_in_bounds(coords))
    }

    fn shape(&self) -> Self::Coordinates {
        (self.rows, self.columns)
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let (row, col) = pos;

        match comptime!((self.config.check_row_bounds, self.config.check_col_bounds)) {
            (true, true) => row < self.rows && col < self.columns,
            (true, false) => row < self.rows,
            (false, true) => col < self.columns,
            (false, false) => true,
        }
    }
}

/// Global layout that uses the last two dimensions and ignores all others.
#[derive(CubeType, Clone, Copy)]
pub struct SimpleTmaGlobalLayout {
    nth_batch: u32,
    #[cube(comptime)]
    transposed: bool,
}

#[cube]
impl SimpleTmaGlobalLayout {
    /// Creates a new 2D layout with the batch set to `nth_batch`.
    pub fn new(nth_batch: u32, #[comptime] layout: MatrixLayout) -> Self {
        let transposed = comptime![matches!(layout, MatrixLayout::ColMajor)];
        SimpleTmaGlobalLayout {
            nth_batch,
            transposed,
        }
    }
}

#[cube]
impl Layout for SimpleTmaGlobalLayout {
    type Coordinates = Coords2d;
    type SourceCoordinates = Coords3d;

    fn to_source_pos(&self, coords: Self::Coordinates) -> Coords3d {
        let (row, col) = coords;
        // Tensor maps are required to have a stride of 1 on the last dim, so their shape is
        // transposed for col-major matrices. Need to compensate by swapping the coordinates.
        if comptime![self.transposed] {
            (self.nth_batch, col, row)
        } else {
            (self.nth_batch, row, col)
        }
    }

    fn to_source_pos_checked(&self, coords: Self::Coordinates) -> (Coords3d, bool) {
        (self.to_source_pos(coords), self.is_in_bounds(coords))
    }

    fn shape(&self) -> Self::Coordinates {
        // No need to bounds check TMA loads
        (u32::MAX, u32::MAX).runtime()
    }

    fn is_in_bounds(&self, _pos: Self::Coordinates) -> bool {
        // No need to bounds check TMA loads
        true.runtime()
    }
}
