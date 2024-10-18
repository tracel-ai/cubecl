use super::TensorView;
use crate::matmul::config::PlaneMapper;
use crate::matmul::matmul_global::GmmConfig;
use crate::matmul::matrix::Ident;
use crate::matmul::cmma_matmul::global::write_coalesced;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

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
pub(crate) fn unload_shared_memory<EG: Numeric, ES: Numeric, G: GmmConfig>(
    view: &mut TensorView<EG>,
    smem_slice: &Slice<'_, Line<ES>>,
    row_tile_begin: u32,
    col_tile_begin: u32,
    #[comptime] config: G
) {
    let stage_dim = config.stage_dim(Ident::Out);
    let slice_line_size = config.out_smem_line_size();
    let out_line_size = config.line_size(Ident::Out);

    let unit_jump = config.plane_dim() * view.tensor.line_size();
    let num_unit_writes = stage_dim.tile_num_elements() / unit_jump;

    let _ = comptime!(check_line_size(out_line_size, slice_line_size));

    for i in 0..num_unit_writes {
        let unit_write = SimpleSmemUnloader::plane_unit() * out_line_size + i * unit_jump;

        let row = row_tile_begin + unit_write / stage_dim.tile_size_y;
        let col = col_tile_begin + unit_write % stage_dim.tile_size_y;

        let value = smem_slice[unit_write / out_line_size];
        write_coalesced::<EG, ES>(view, row, col, value);
    }
}

fn check_line_size(out_line_size: u32, slice_line_size: u32) {
    assert_eq!(out_line_size, slice_line_size, 
        "Error: Expected global output and output shared memory to have equal line size, but found out_line_size = {} and slice_line_size = {}.",
        out_line_size, slice_line_size
    );
}
