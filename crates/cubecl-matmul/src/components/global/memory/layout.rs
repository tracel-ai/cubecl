use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::tensor::r#virtual::VirtualTensor;

use crate::components::{
    MatrixLayout,
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
    layout: MatrixLayout,
}

#[cube]
impl SimpleGlobalLayout {
    pub fn new<T: Numeric>(tensor: &VirtualTensor<T>, #[comptime] layout: MatrixLayout) -> Self {
        let rank = tensor.rank();
        match layout {
            MatrixLayout::RowMajor => SimpleGlobalLayout {
                rows: tensor.shape(rank - 2),
                stride_row: tensor.stride(rank - 2),
                columns: tensor.shape(rank - 1),
                stride_col: tensor.stride(rank - 1),
                layout,
            },
            MatrixLayout::ColMajor => SimpleGlobalLayout {
                rows: tensor.shape(rank - 1),
                stride_row: tensor.stride(rank - 1),
                columns: tensor.shape(rank - 2),
                stride_col: tensor.stride(rank - 2),
                layout,
            },
        }
    }
}

#[cube]
impl Layout for SimpleGlobalLayout {
    type Coordinates = Coords2d;

    fn to_linear(this: &Self, coords: Self::Coordinates) -> u32 {
        coords.0 * this.stride_row + coords.1 * this.stride_col
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_linear(this: &Self, idx: u32) -> Self::Coordinates {
        match comptime!(this.layout) {
            MatrixLayout::RowMajor => {
                let col = idx % this.columns;
                let row = (idx / this.columns) % this.rows;
                (row, col)
            }
            MatrixLayout::ColMajor => {
                let row = idx % this.rows;
                let col = (idx / this.rows) % this.columns;
                (row, col)
            }
        }
    }

    fn shape(this: &Self) -> Self::Coordinates {
        (this.rows, this.columns)
    }

    fn offset(
        _this: &Self,
        coords: Self::Coordinates,
        offset: Self::Coordinates,
    ) -> Self::Coordinates {
        (coords.0 + offset.0, coords.1 + offset.1)
    }
}
