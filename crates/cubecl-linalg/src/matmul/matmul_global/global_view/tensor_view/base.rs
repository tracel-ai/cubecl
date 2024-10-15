use crate::matmul::matmul_global::global_view::tensor_view::smem2tensor::{
    Smem2Tensor, Smem2TensorSimple,
};
use crate::matmul::matmul_global::GlobalView;
use crate::matmul::matmul_stage::{Gmem2SmemContinuous, SharedMemoryLoader, TilingOrder};
use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::stage_info::StageInfo;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
pub struct TensorView<E: Numeric> {
    pub tensor: Tensor<Line<E>>,
    pub layout: MatrixLayout,
    pub x_offset: u32,
    pub y_offset: u32,
    pub stride_x: u32,
    pub stride_y: u32,
    pub shape_x: u32,
    pub shape_y: u32,
}

#[cube]
impl<EG: Numeric> GlobalView<EG> for TensorView<EG> {
    type Global = Tensor<Line<EG>>;

    fn load_coalesced(
        view: &Self,
        tile_x: u32,
        tile_y: u32,
        load_id: u32,
        tile_size_x: u32,
        tile_size_y: u32,
    ) -> Line<EG> {
        let tensor = &view.tensor;

        let view_tile_x = tile_x * tile_size_x + view.x_offset;
        let view_tile_y = tile_y * tile_size_y + view.y_offset;

        let (load_x, load_y) = match comptime!(view.layout) {
            MatrixLayout::RowMajor => (load_id / tile_size_y, load_id % tile_size_y),
            MatrixLayout::ColMajor => (load_id % tile_size_x, load_id / tile_size_x),
        };

        let view_x = view_tile_x + load_x;
        let view_y = view_tile_y + load_y;

        let read_pos = (view_x * view.stride_x + view_y * view.stride_y) / tensor.line_size();

        select(
            view_x < view.shape_x && view_y < view.shape_y,
            tensor[read_pos],
            Line::empty(tensor.line_size()).fill(EG::from_int(0)),
        )
    }

    fn load_shared_memory<ES: Numeric, O: TilingOrder>(
        view: &Self,
        shared_memory: &mut SharedMemory<Line<ES>>,
        #[comptime] stage_info: StageInfo,
    ) {
        // TODO allow other modes than Gmem2SmemContinuous
        Gmem2SmemContinuous::load_shared_memory::<EG, ES, Self, O>(view, shared_memory, stage_info);
    }

    fn init_view(view: &mut Self, x_offset: u32, y_offset: u32) {
        view.x_offset = x_offset;
        view.y_offset = y_offset;
    }

    fn update_view(view: &mut Self, x_offset: u32, y_offset: u32) {
        // TODO in practice one of them is always += 0, so there is useless computation
        view.x_offset += x_offset;
        view.y_offset += y_offset;
    }

    /// Assumes (write_row, write_col) is within bounds
    /// Does not account for batch offset
    /// Assumes out is row major
    fn write_coalesced<ES: Numeric>(view: &mut Self, write_x: u32, write_y: u32, value: Line<ES>) {
        let tensor = &mut view.tensor;
        let write_row = write_x + view.x_offset;
        let write_col = write_y + view.y_offset;

        let write_position =
            (write_row * view.stride_x + write_col * view.stride_y) / tensor.line_size();
        tensor[write_position] = Line::cast_from(value);

        // TODO ternary to avoid branching
        if write_x >= view.shape_x || write_y >= view.shape_y {
        } else {
            tensor[write_position] = Line::cast_from(value);
        }
    }

    fn write_slice<ES: Numeric>(
        view: &mut Self,
        slice: &Slice<'_, Line<ES>>,
        write_x: u32,
        write_y: u32,
        #[comptime] stage_info: StageInfo,
    ) {
        Smem2TensorSimple::smem_to_tensor(view, slice, write_x, write_y, stage_info);
    }
}

#[cube]
pub fn new_tensor_view<E: Numeric>(tensor: Tensor<Line<E>>, layout: MatrixLayout) -> TensorView<E> {
    let rank = tensor.rank();
    let stride_x = tensor.stride(rank - 2);
    let stride_y = tensor.stride(rank - 1);
    let shape_x = tensor.shape(rank - 2);
    let shape_y = tensor.shape(rank - 1);

    TensorView::<E> {
        tensor,
        layout,
        x_offset: 0,
        y_offset: 0,
        stride_x,
        stride_y,
        shape_x,
        shape_y,
    }
}
