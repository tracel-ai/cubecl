use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::matrix_layout::MatrixLayout;

use super::GlobalView;

#[derive(CubeType)]
pub struct ArrayView<E: Numeric> {
    pub array: Array<Line<E>>,
    pub layout: MatrixLayout,
    pub shape: (u32, u32),
}

#[cube]
impl<E: Numeric> GlobalView<E> for ArrayView<E> {
    type Global = Array<Line<E>>;

    fn load_single(view: &Self, read_row: u32, read_col: u32) -> Line<E> {
        let array = &view.array;
        let (stride_row, stride_col) = match comptime!(view.layout) {
            MatrixLayout::RowMajor => (view.shape.1, 1),
            MatrixLayout::ColMajor => (1, view.shape.0),
        };

        let read_pos = (read_row * stride_row + read_col * stride_col) / array.line_size();

        array[read_pos]
    }

    fn init_view(_view: &mut Self, _x_offset: u32, _y_offset: u32) {
        // ArrayView does not support offsets
    }

    fn update_view(_view: &mut Self, _x_offset: u32, _y_offset: u32) {
        // ArrayView does not support offsets
    }
}

#[cube]
pub fn new_array_view<E: Numeric>(
    array: Array<Line<E>>,
    layout: MatrixLayout,
    shape: (u32, u32),
) -> ArrayView<E> {
    ArrayView::<E> {
        array,
        layout,
        shape,
    }
}
