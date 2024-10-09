use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::matrix_layout::MatrixLayout;

#[cube]
pub trait GmemView<E: Numeric>: CubeType {
    type Gmem: CubeType;

    fn load_single(view: &Self, read_row: u32, read_col: u32) -> Line<E>;
    fn update_view(view: &mut Self, x_offset: u32, y_offset: u32);
}

#[derive(CubeType)]
pub struct TensorView<E: Numeric> {
    pub tensor: Tensor<Line<E>>,
    pub layout: MatrixLayout,
    pub x_offset: u32,
    pub y_offset: u32,
}

#[derive(CubeType)]
pub struct ArrayView<E: Numeric> {
    pub array: Array<Line<E>>,
    pub layout: MatrixLayout,
    pub shape: (u32, u32),
}

#[cube]
impl<E: Numeric> GmemView<E> for TensorView<E> {
    type Gmem = Tensor<Line<E>>;

    fn load_single(view: &TensorView<E>, read_row: u32, read_col: u32) -> Line<E> {
        let tensor = &view.tensor;
        let read_row = read_row + view.x_offset;
        let read_col = read_col + view.y_offset;

        let read_pos = (read_row * tensor.stride(tensor.rank() - 2)
            + read_col * tensor.stride(tensor.rank() - 1))
            / tensor.line_size();

        tensor[read_pos]
    }

    fn update_view(view: &mut Self, x_offset: u32, y_offset: u32) {
        view.x_offset = x_offset;
        view.y_offset = y_offset;
    }
}

#[cube]
pub fn new_tensor_view<E: Numeric>(tensor: Tensor<Line<E>>, layout: MatrixLayout) -> TensorView<E> {
    TensorView::<E> {
        tensor,
        layout,
        x_offset: 0,
        y_offset: 0,
    }
}

#[cube]
impl<E: Numeric> GmemView<E> for ArrayView<E> {
    type Gmem = Array<Line<E>>;
    fn load_single(view: &ArrayView<E>, read_row: u32, read_col: u32) -> Line<E> {
        let array = &view.array;
        let (stride_row, stride_col) = match comptime!(view.layout) {
            MatrixLayout::RowMajor => (view.shape.1, 1),
            MatrixLayout::ColMajor => (1, view.shape.0),
        };

        let read_pos = (read_row * stride_row + read_col * stride_col) / array.line_size();

        array[read_pos]
    }

    fn update_view(_view: &mut Self, _x_offset: u32, _y_offset: u32) {
        // ArrayView does not support update
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
