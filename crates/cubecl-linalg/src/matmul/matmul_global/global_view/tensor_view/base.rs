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

        let absolute_tile_x = tile_x * tile_size_x + view.x_offset;
        let absolute_tile_y = tile_y * tile_size_y + view.y_offset;

        let (load_x, load_y) = match comptime!(view.layout) {
            MatrixLayout::RowMajor => (load_id / tile_size_y, load_id % tile_size_y),
            MatrixLayout::ColMajor => (load_id % tile_size_x, load_id / tile_size_x),
        };

        let read_pos = ((absolute_tile_x + load_x) * view.stride_x
            + (absolute_tile_y + load_y) * view.stride_y)
            / tensor.line_size();

        tensor[read_pos]
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
        view.x_offset += x_offset;
        view.y_offset += y_offset;
    }

    /// Assumes (write_row, write_col) is within bounds
    /// Does not account for batch offset
    /// Assumes out is row major
    fn write_coalesced<ES: Numeric>(
        view: &mut Self,
        write_row: u32,
        write_col: u32,
        value: Line<ES>,
    ) {
        let tensor = &mut view.tensor;
        let write_row = write_row + view.x_offset;
        let write_col = write_col + view.y_offset;

        let write_position =
            (write_row * view.stride_x + write_col * view.stride_y) / tensor.line_size();
        tensor[write_position] = Line::cast_from(value);
    }

    fn write_slice<ES: Numeric>(
        view: &mut Self,
        slice: &Slice<'_, Line<ES>>,
        write_row: u32,
        write_col: u32,
        #[comptime] stage_info: StageInfo,
    ) {
        Smem2TensorSimple::smem_to_tensor(view, slice, write_row, write_col, stage_info);
    }
}

#[cube]
pub fn new_tensor_view<E: Numeric>(tensor: Tensor<Line<E>>, layout: MatrixLayout) -> TensorView<E> {
    let stride_x = tensor.stride(tensor.rank() - 2);
    let stride_y = tensor.stride(tensor.rank() - 1);
    TensorView::<E> {
        tensor,
        layout,
        x_offset: 0,
        y_offset: 0,
        stride_x,
        stride_y,
    }
}
