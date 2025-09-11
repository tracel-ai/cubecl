use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::tensor::{
    layout::{Coords1d, Coords3d, Layout, LayoutExpand},
    r#virtual::VirtualTensor,
};

use crate::components::global::memory::GlobalMemoryConfig;

/// Global layout that uses the last two dimensions and ignores all others.
/// Batch dim is treated as unit stride, and batch shape is always `1`
#[derive(CubeType, Clone, Copy)]
pub struct SimpleGlobalLayout {
    rows: u32,
    stride_row: u32,
    columns: u32,
    stride_col: u32,
    #[cube(comptime)]
    config: GlobalMemoryConfig,
}

#[cube]
impl SimpleGlobalLayout {
    pub fn new<T: Numeric, IO: Clone>(
        tensor: &VirtualTensor<T, IO>,
        #[comptime] config: GlobalMemoryConfig,
    ) -> Self {
        let rank = tensor.rank();

        SimpleGlobalLayout {
            rows: tensor.shape(rank - 2),
            stride_row: tensor.stride(rank - 2),
            columns: tensor.shape(rank - 1),
            stride_col: tensor.stride(rank - 1),
            config,
        }
    }
}

#[cube]
impl Layout for SimpleGlobalLayout {
    type Coordinates = Coords3d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, coords: Self::Coordinates) -> u32 {
        let (b, row, col) = coords;
        let idx = b + row * self.stride_row + col * self.stride_col;
        idx / comptime![self.config.global_line_size]
    }

    fn to_source_pos_checked(&self, coords: Self::Coordinates) -> (u32, bool) {
        (self.to_source_pos(coords), self.is_in_bounds(coords))
    }

    fn shape(&self) -> Self::Coordinates {
        (1, self.rows, self.columns)
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let (_, row, col) = pos;

        match comptime!((self.config.check_row_bounds, self.config.check_col_bounds)) {
            (true, true) => row < self.rows && col < self.columns,
            (true, false) => row < self.rows,
            (false, true) => col < self.columns,
            (false, false) => true,
        }
    }
}
