use crate::matmul::matmul_global::global_view::tensor_view::smem2tensor::{
    Smem2Tensor, Smem2TensorSimple,
};
use crate::matmul::matmul_global::GlobalView;
use crate::matmul::matmul_stage::{Gmem2SmemContinuous, RowMajorTiling, SharedMemoryLoader};
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
}

#[cube]
impl<E: Numeric> GlobalView<E> for TensorView<E> {
    type Global = Tensor<Line<E>>;

    fn line_size(view: &Self) -> u32 {
        comptime!(view.tensor.line_size())
    }

    fn load_single(view: &Self, read_row: u32, read_col: u32) -> Line<E> {
        let tensor = &view.tensor;
        let read_row = read_row + view.x_offset;
        let read_col = read_col + view.y_offset;

        // TODO stride computations should be done once in the new
        let read_pos = (read_row * tensor.stride(tensor.rank() - 2)
            + read_col * tensor.stride(tensor.rank() - 1))
            / tensor.line_size();

        tensor[read_pos]
    }

    fn load_shared_memory<ES: Numeric>(
        view: &Self,
        shared_memory: &mut SharedMemory<Line<ES>>,
        #[comptime] stage_info: StageInfo,
    ) {
        // TODO allow other modes / tilings
        Gmem2SmemContinuous::load_shared_memory::<E, ES, Self, RowMajorTiling>(
            view,
            shared_memory,
            stage_info,
        );
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
    fn write_single<C: CubePrimitive>(view: &mut Self, write_row: u32, write_col: u32, value: C) {
        let tensor = &mut view.tensor;
        let write_row = write_row + view.x_offset;
        let write_col = write_col + view.y_offset;

        // TODO stride computations should be done once in the new
        let write_position = (write_row * tensor.stride(tensor.rank() - 2)
            + write_col * tensor.stride(tensor.rank() - 1))
            / tensor.line_size();
        tensor[write_position] = Line::cast_from(value);
    }

    fn write_slice<C: CubePrimitive>(
        view: &mut Self,
        slice: &Slice<'_, C>,
        write_row: u32,
        write_col: u32,
        #[comptime] stage_info: StageInfo,
    ) {
        Smem2TensorSimple::smem_to_tensor(view, slice, write_row, write_col, stage_info);
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
