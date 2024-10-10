use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::matmul_global::{smem_slice_to_gmem, GlobalView};
use crate::matmul::matmul_stage::{Gmem2SmemContinuous, RowMajorTiling, SharedMemoryLoader};
use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::stage_info::StageInfo;

#[derive(CubeType)]
pub struct ArrayView<E: Numeric> {
    pub array: Array<Line<E>>,
    pub layout: MatrixLayout,
    pub shape: (u32, u32),
}

#[cube]
impl<E: Numeric> GlobalView<E> for ArrayView<E> {
    type Global = Array<Line<E>>;

    fn line_size(view: &Self) -> u32 {
        comptime!(view.array.line_size())
    }

    fn load_single(view: &Self, read_row: u32, read_col: u32) -> Line<E> {
        let array = &view.array;

        // TODO stride computations should be done once in the new
        let (stride_row, stride_col) = match comptime!(view.layout) {
            MatrixLayout::RowMajor => (view.shape.1, 1),
            MatrixLayout::ColMajor => (1, view.shape.0),
        };

        let read_pos = (read_row * stride_row + read_col * stride_col) / array.line_size();

        array[read_pos]
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

    fn init_view(_view: &mut Self, _x_offset: u32, _y_offset: u32) {
        // ArrayView does not support offsets
    }

    fn update_view(_view: &mut Self, _x_offset: u32, _y_offset: u32) {
        // ArrayView does not support offsets
    }

    fn write_single<C: CubePrimitive>(view: &mut Self, write_row: u32, write_col: u32, value: C) {
        let array = &mut view.array;

        // TODO stride computations should be done once in the new
        let (stride_row, stride_col) = match comptime!(view.layout) {
            MatrixLayout::RowMajor => (view.shape.1, 1),
            MatrixLayout::ColMajor => (1, view.shape.0),
        };

        let write_pos = (write_row * stride_row + write_col * stride_col) / array.line_size();

        array[write_pos] = Line::cast_from(value);
    }

    fn write_slice<C: CubePrimitive>(
        view: &mut Self,
        slice: &Slice<'_, C>,
        write_row: u32,
        write_col: u32,
        #[comptime] stage_info: StageInfo,
    ) {
        smem_slice_to_gmem(view, slice, write_row, write_col, stage_info);
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
