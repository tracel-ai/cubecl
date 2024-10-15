use super::TensorView;
use crate::matmul::config::PlaneMapper;
use crate::matmul::matmul_global::GlobalView;
use crate::matmul::stage_info::{tile_num_elements, StageInfo};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub trait Smem2Tensor {
    fn smem_to_tensor<E: Numeric, ES: Numeric>(
        out: &mut TensorView<E>,
        smem_slice: &Slice<'_, Line<ES>>,
        compute_plane_offset: u32,
        accumulator_offset: u32,
        #[comptime] stage_info: StageInfo,
        #[comptime] slice_line_size: u32,
    );
}

#[derive(CubeType)]
pub struct Smem2TensorSimple {}

#[cube]
impl PlaneMapper for Smem2TensorSimple {
    fn plane_id() -> u32 {
        UNIT_POS_Y
    }

    fn plane_unit() -> u32 {
        UNIT_POS_X
    }

    fn num_planes() -> u32 {
        CUBE_DIM_Y
    }

    fn plane_dim() -> u32 {
        CUBE_DIM_X
    }
}

#[cube]
impl Smem2Tensor for Smem2TensorSimple {
    fn smem_to_tensor<E: Numeric, ES: Numeric>(
        out: &mut TensorView<E>,
        slice: &Slice<'_, Line<ES>>,
        row_tile_begin: u32,
        col_tile_begin: u32,
        #[comptime] stage_info: StageInfo,
        #[comptime] slice_line_size: u32,
    ) {
        let unit_jump = Self::plane_dim() * out.tensor.line_size();
        let num_unit_writes = tile_num_elements(stage_info) / unit_jump;
        let out_line_size = out.tensor.line_size();

        for i in 0..num_unit_writes {
            let unit_write = Self::plane_unit() * out_line_size + i * unit_jump;

            let row = row_tile_begin + unit_write / stage_info.tile_size_y;
            let col = col_tile_begin + unit_write % stage_info.tile_size_y;

            // if comptime!(out_line_size == slice_line_size) {
            let value = slice[unit_write / slice_line_size];
            TensorView::write_coalesced::<ES>(out, row, col, value);
            // }
        }
    }
}
