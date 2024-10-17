use super::TensorView;
use crate::matmul::cmma_matmul::config::CmmaConfig;
use crate::matmul::config::{MatmulConfig, PlaneMapper};
use crate::matmul::matmul_global::{GmmConfig, WriteView};
use crate::matmul::matrix::Ident;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub trait SharedMemoryUnloader {
    type Config: GmmConfig;

    fn unload_shared_memory<E: Numeric, ES: Numeric>(
        out: &mut TensorView<E>,
        smem_slice: &Slice<'_, Line<ES>>,
        compute_plane_offset: u32,
        accumulator_offset: u32,
        #[comptime] config: Self::Config
    );
}

#[derive(CubeType)]
pub struct SimpleSmemUnloader {}

#[cube]
impl PlaneMapper for SimpleSmemUnloader {
    fn plane_id() -> u32 {
        UNIT_POS_Y
    }

    fn plane_unit() -> u32 {
        UNIT_POS_X
    }
}

#[cube]
impl SharedMemoryUnloader for SimpleSmemUnloader {
    type Config = CmmaConfig;

    fn unload_shared_memory<E: Numeric, ES: Numeric>(
        out: &mut TensorView<E>,
        smem_slice: &Slice<'_, Line<ES>>,
        row_tile_begin: u32,
        col_tile_begin: u32,
        #[comptime] config: Self::Config
    ) {
        let stage_dim = config.stage_dim(Ident::Out);
        let slice_line_size = config.out_smem_line_size;
        let out_line_size = config.line_size(Ident::Out);

        let unit_jump = config.plane_dim() * out.tensor.line_size();
        let num_unit_writes = stage_dim.tile_num_elements() / unit_jump;

        let _ = comptime!(check_line_size(out_line_size, slice_line_size));

        for i in 0..num_unit_writes {
            let unit_write = Self::plane_unit() * out_line_size + i * unit_jump;

            let row = row_tile_begin + unit_write / stage_dim.tile_size_y;
            let col = col_tile_begin + unit_write % stage_dim.tile_size_y;

            let value = smem_slice[unit_write / out_line_size];
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
