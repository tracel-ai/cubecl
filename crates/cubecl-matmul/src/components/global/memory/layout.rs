use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::tensor::r#virtual::VirtualTensor;

use crate::components::{
    global::memory::GlobalMemoryConfig,
    layout::{Coords2d, Layout},
};

/// Global layout that uses the last two dimensions and ignores all others
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
    pub fn new<T: Numeric>(
        tensor: &VirtualTensor<T>,
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
    type Coordinates = Coords2d;

    fn to_linear(this: &Self, coords: Self::Coordinates) -> u32 {
        coords.0 * this.stride_row + coords.1 * this.stride_col
    }

    fn to_linear_checked(this: &Self, coords: Self::Coordinates) -> (u32, bool) {
        let (row, col) = coords;
        let idx = row * this.stride_row + col * this.stride_col;

        let in_bounds =
            match comptime!((this.config.check_row_bounds, this.config.check_col_bounds)) {
                (true, true) => row < this.rows && col < this.columns,
                (true, false) => row < this.rows,
                (false, true) => col < this.columns,
                (false, false) => true,
            };

        (idx, in_bounds)
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_linear(this: &Self, idx: u32) -> Self::Coordinates {
        let row = (idx / this.stride_row) % this.rows;
        let col = (idx / this.stride_col) % this.columns;
        (row, col)
    }

    fn shape(this: &Self) -> Self::Coordinates {
        (this.rows, this.columns)
    }
}
