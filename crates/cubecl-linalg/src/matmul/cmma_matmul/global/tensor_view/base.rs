use crate::matmul::cmma_matmul::global::load_shared_memory;
use crate::matmul::cmma_matmul::global::unload_shared_memory;
use crate::matmul::matmul_global::GmmConfig;
use crate::matmul::matrix::{Ident, MatrixLayout};
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
pub(crate) fn load_coalesced<EG: Numeric, G: GmmConfig>(
    this: &TensorView<EG>,
    tile_x: u32,
    tile_y: u32,
    load_id: u32,
    #[comptime] ident: Ident,
    #[comptime] config: G,
) -> Line<EG> {
    let tensor = &this.tensor;
    let line_size = config.line_size(ident);
    let tile_size_x = config.stage_dim(ident).tile_size_x;
    let tile_size_y = config.stage_dim(ident).tile_size_y;

    let view_tile_x = tile_x * tile_size_x + this.x_offset;
    let view_tile_y = tile_y * tile_size_y + this.y_offset;

    let (load_x, load_y) = match config.layout(ident) {
        MatrixLayout::RowMajor => (load_id / tile_size_y, load_id % tile_size_y),
        MatrixLayout::ColMajor => (load_id % tile_size_x, load_id / tile_size_x),
    };

    let view_x = view_tile_x + load_x;
    let view_y = view_tile_y + load_y;

    let read_pos = (view_x * this.stride_x + view_y * this.stride_y) / line_size;

    select(
        view_x < this.shape_x && view_y < this.shape_y,
        tensor[read_pos],
        Line::empty(line_size).fill(EG::from_int(0)),
    )
}

#[cube]
// TODO rename
pub(crate) fn load_shared_memory_<EG: Numeric, ES: Numeric, G: GmmConfig>(
    view: &TensorView<EG>,
    shared_memory: &mut SharedMemory<Line<ES>>,
    #[comptime] ident: Ident,
    #[comptime] config: G,
) {
    // TODO allow other modes than Gmem2SmemContinuous
    // TODO allow YMAjor
    load_shared_memory::<EG, ES, G>(view, shared_memory, ident, config);
}

#[cube]
pub(crate) fn init_view<EG: Numeric>(view: &mut TensorView<EG>, x_offset: u32, y_offset: u32) {
    view.x_offset = x_offset;
    view.y_offset = y_offset;
}

#[cube]
pub(crate) fn update_view<EG: Numeric>(view: &mut TensorView<EG>, x_offset: u32, y_offset: u32) {
    // TODO in practice one of them is always += 0, so there is useless computation
    // With ident
    view.x_offset += x_offset;
    view.y_offset += y_offset;
}

#[cube]
pub(crate) fn write_coalesced<EG: Numeric, ES: Numeric>(
    view: &mut TensorView<EG>,
    write_x: u32,
    write_y: u32,
    value: Line<ES>,
) {
    let tensor = &mut view.tensor;
    let view_x = write_x + view.x_offset;
    let view_y = write_y + view.y_offset;

    let write_position = (view_x * view.stride_x + view_y * view.stride_y) / tensor.line_size();

    // TODO: will need comptime checkbound condition because we can't use select for not writing
    if write_x < view.shape_x && write_y < view.shape_y {
        tensor[write_position] = Line::cast_from(value);
    }
}

#[cube]
pub(crate) fn write_slice<EG: Numeric, ES: Numeric, G: GmmConfig>(
    view: &mut TensorView<EG>,
    slice: &Slice<'_, Line<ES>>,
    write_x: u32,
    write_y: u32,
    #[comptime] config: G,
) {
    unload_shared_memory::<EG, ES, G>(view, slice, write_x, write_y, config);
}

#[cube]
pub fn new_tensor_view<E: Numeric>(tensor: Tensor<Line<E>>) -> TensorView<E> {
    let rank = tensor.rank();
    let stride_x = tensor.stride(rank - 2);
    let stride_y = tensor.stride(rank - 1);
    let shape_x = tensor.shape(rank - 2);
    let shape_y = tensor.shape(rank - 1);

    TensorView::<E> {
        tensor,
        x_offset: 0,
        y_offset: 0,
        stride_x,
        stride_y,
        shape_x,
        shape_y,
    }
}
