use super::TensorView;
use crate::matmul::cmma_matmul::config::CmmaConfig;
use crate::matmul::config::{MatmulConfig, PlaneMapper};
use crate::matmul::matmul_global::{GlobalView, GmmConfig};
use crate::matmul::stage_info::{tile_num_elements, StageInfo};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub trait Smem2Tensor {
    type Config: GmmConfig;

    fn smem_to_tensor<E: Numeric, ES: Numeric>(
        out: &mut TensorView<E>,
        smem_slice: &Slice<'_, Line<ES>>,
        compute_plane_offset: u32,
        accumulator_offset: u32,
        #[comptime] stage_info: StageInfo,
        #[comptime] slice_line_size: u32,
        #[comptime] config: Self::Config
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
}

#[cube]
impl Smem2Tensor for Smem2TensorSimple {
    type Config = CmmaConfig;

    fn smem_to_tensor<E: Numeric, ES: Numeric>(
        out: &mut TensorView<E>,
        slice: &Slice<'_, Line<ES>>,
        row_tile_begin: u32,
        col_tile_begin: u32,
        #[comptime] stage_info: StageInfo,
        #[comptime] slice_line_size: u32,
        #[comptime] config: Self::Config
    ) {
        let unit_jump = config.plane_dim() * out.tensor.line_size();
        let num_unit_writes = tile_num_elements(stage_info) / unit_jump;
        let out_line_size = out.tensor.line_size();

        let _ = comptime!(check_line_size(out_line_size, slice_line_size));

        for i in 0..num_unit_writes {
            let unit_write = Self::plane_unit() * out_line_size + i * unit_jump;

            let row = row_tile_begin + unit_write / stage_info.tile_size_y;
            let col = col_tile_begin + unit_write % stage_info.tile_size_y;

            let value = slice[unit_write / out_line_size];
            TensorView::write_coalesced::<ES>(out, row, col, value);
        }
    }
}

fn check_line_size(out_line_size: u32, slice_line_size: u32) {
    assert_eq!(out_line_size, slice_line_size, 
        "Error: Expected global output and output shared memory to have equal line size, but found out_line_size = {} and slice_line_size = {}.",
        out_line_size, slice_line_size
    );
    
}
