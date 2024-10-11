use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::matmul_global::{smem_slice_to_gmem, GlobalView};
use crate::matmul::matmul_stage::{Gmem2SmemContinuous, SharedMemoryLoader, TilingOrder};
use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::stage_info::StageInfo;

#[derive(CubeType)]
pub struct ArrayView<E: Numeric> {
    pub array: Array<Line<E>>,
    pub layout: MatrixLayout,
    pub shape: (u32, u32),
    pub stride_x: u32,
    pub stride_y: u32,
}

#[cube]
impl<E: Numeric> GlobalView<E> for ArrayView<E> {
    type Global = Array<Line<E>>;

    fn line_size(view: &Self) -> u32 {
        comptime!(view.array.line_size())
    }

    fn load_coalesced(
        view: &Self,
        tile_x: u32,
        tile_y: u32,
        load_id: u32,
        tile_size_x: u32,
        tile_size_y: u32,
    ) -> Line<E> {
        let array = &view.array;

        let (load_x, load_y) = match comptime!(view.layout) {
            MatrixLayout::RowMajor => (load_id / tile_size_y, load_id % tile_size_y),
            MatrixLayout::ColMajor => (load_id % tile_size_x, load_id / tile_size_x),
        };

        let read_pos = ((tile_x * tile_size_x + load_x) * view.stride_x
            + (tile_y * tile_size_y + load_y) * view.stride_y)
            / array.line_size();

        array[read_pos]
    }

    fn load_shared_memory<ES: Numeric, O: TilingOrder>(
        view: &Self,
        shared_memory: &mut SharedMemory<Line<ES>>,
        #[comptime] stage_info: StageInfo,
    ) {
        // TODO allow other modes than Gmem2SmemContinuous
        Gmem2SmemContinuous::load_shared_memory::<E, ES, Self, O>(view, shared_memory, stage_info);
    }

    fn init_view(_view: &mut Self, _x_offset: u32, _y_offset: u32) {
        // ArrayView does not support offsets
    }

    fn update_view(_view: &mut Self, _x_offset: u32, _y_offset: u32) {
        // ArrayView does not support offsets
    }

    fn write_coalesced<C: CubePrimitive>(view: &mut Self, write_x: u32, write_y: u32, value: C) {
        let array = &mut view.array;

        let write_pos = (write_x * view.stride_x + write_y * view.stride_y) / array.line_size();

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
    let (stride_x, stride_y) = match comptime!(layout) {
        MatrixLayout::RowMajor => (shape.1, 1),
        MatrixLayout::ColMajor => (1, shape.0),
    };
    ArrayView::<E> {
        array,
        layout,
        shape,
        stride_x,
        stride_y,
    }
}
