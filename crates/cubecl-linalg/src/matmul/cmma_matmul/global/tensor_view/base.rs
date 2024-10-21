use crate::matmul::matrix::Ident;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
pub struct TensorView<E: Numeric> {
    pub tensor: Tensor<Line<E>>,
    pub x_offset: u32,
    pub y_offset: u32,
    pub stride_x: u32,
    pub stride_y: u32,
    pub shape_x: u32,
    pub shape_y: u32,
}

#[cube]
pub(crate) fn update_view<EG: Numeric>(
    view: &mut TensorView<EG>,
    x_offset: u32,
    y_offset: u32,
    #[comptime] ident: Ident,
) {
    match ident {
        Ident::Lhs => {
            view.y_offset += y_offset;
        }
        Ident::Rhs => {
            view.x_offset += x_offset;
        }
        Ident::Out => {}
    }
}

#[cube]
pub fn new_tensor_view<E: Numeric>(
    tensor: Tensor<Line<E>>,
    x_offset: u32,
    y_offset: u32,
) -> TensorView<E> {
    let rank = tensor.rank();
    let stride_x = tensor.stride(rank - 2);
    let stride_y = tensor.stride(rank - 1);
    let shape_x = tensor.shape(rank - 2);
    let shape_y = tensor.shape(rank - 1);

    TensorView::<E> {
        tensor,
        x_offset,
        y_offset,
        stride_x,
        stride_y,
        shape_x,
        shape_y,
    }
}
