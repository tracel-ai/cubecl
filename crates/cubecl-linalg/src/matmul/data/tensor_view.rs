use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::matrix_layout::MatrixLayout;

#[cube]
pub trait GmemView<E: Numeric>: CubeType {
    type Gmem: CubeType;

    fn load_single(view: &Self, read_row: u32, read_col: u32) -> Line<E>;
}

#[derive(CubeType)]
pub struct TensorView<E: Numeric> {
    pub tensor: Tensor<Line<E>>,
    pub layout: MatrixLayout,
    pub x_offset: u32,
    pub y_offset: u32,
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
}
