use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::matrix_layout::MatrixLayout;

use super::GlobalView;

#[derive(CubeType)]
pub struct TensorView<E: Numeric> {
    pub tensor: Tensor<Line<E>>,
    pub layout: MatrixLayout,
    pub x_offset: u32,
    pub y_offset: u32,
}

#[cube]
impl<E: Numeric> GlobalView<E> for TensorView<E> {
    type Global = Tensor<Line<E>>;

    fn load_single(view: &Self, read_row: u32, read_col: u32) -> Line<E> {
        let tensor = &view.tensor;
        let read_row = read_row + view.x_offset;
        let read_col = read_col + view.y_offset;

        let read_pos = (read_row * tensor.stride(tensor.rank() - 2)
            + read_col * tensor.stride(tensor.rank() - 1))
            / tensor.line_size();

        tensor[read_pos]
    }

    fn init_view(view: &mut Self, x_offset: u32, y_offset: u32) {
        view.x_offset = x_offset;
        view.y_offset = y_offset;
    }

    fn update_view(view: &mut Self, x_offset: u32, y_offset: u32) {
        view.x_offset += x_offset;
        view.y_offset += y_offset;
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
