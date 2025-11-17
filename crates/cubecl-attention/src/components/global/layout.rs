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
    /// Creates a new 2D layout starting at `batch_index`.
    pub fn new<T: Numeric, IO: Clone>(
        tensor: &VirtualTensor<T, IO>,
        batch_index: u32,
        #[comptime] config: GlobalMemoryConfig,
    ) -> Self {
        let stride_batch = tensor.stride(1);
        AttentionGlobalLayout {
            rows: tensor.shape(2),
            stride_row: tensor.stride(2),
            columns: tensor.shape(3),
            stride_col: tensor.stride(3),
            batch_offset: batch_index * stride_batch,
            config,
        }
    }
}

#[cube]
impl Layout for AttentionGlobalLayout {
    type Coordinates = Coords2d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, coords: Self::Coordinates) -> u32 {
        let line_size = comptime![self.config.line_size()];
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

        match comptime!((
            self.config.check_row_bounds(),
            self.config.check_col_bounds()
        )) {
            (true, true) => row < self.rows && col < self.columns,
            (true, false) => row < self.rows,
            (false, true) => col < self.columns,
            (false, false) => true,
        }
    }
}
