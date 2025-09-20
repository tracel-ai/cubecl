use cubecl::prelude::*;
use cubecl_core::{self as cubecl};
use cubecl_matmul::components::global::memory::GlobalMemoryConfig;
use cubecl_std::tensor::{
    layout::{Coords1d, Coords2d, Layout, LayoutExpand},
    r#virtual::VirtualTensor,
};

/// Global layout that uses the last two dimensions and ignores all others.
#[derive(CubeType, Clone, Copy)]
pub struct AttentionGlobalLayout {
    rows: u32,
    stride_row: u32,
    columns: u32,
    stride_col: u32,
    batch_offset: u32,
    #[cube(comptime)]
    config: GlobalMemoryConfig,
}

#[cube]
impl AttentionGlobalLayout {
    /// Creates a new 2D layout starting at `batch_offset`.
    pub fn new<T: Numeric, IO: Clone>(
        tensor: &VirtualTensor<T, IO>,
        batch_offset: u32,
        #[comptime] config: GlobalMemoryConfig,
    ) -> Self {
        AttentionGlobalLayout {
            rows: tensor.shape(1),
            stride_row: tensor.stride(1),
            columns: tensor.shape(3),
            stride_col: tensor.stride(3),
            batch_offset,
            config,
        }
    }
}

#[cube]
impl Layout for AttentionGlobalLayout {
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

    fn to_source_shape(&self, shape: Self::Coordinates) -> Self::SourceCoordinates {
        let size = shape.0 * shape.1;
        size / comptime![self.config.global_line_size]
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
