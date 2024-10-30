use super::TensorView;
use crate::matmul::matmul_modular::config::PlaneMapper;
use crate::matmul::matmul_modular::matmul_global::GmmConfig;
use crate::matmul::matmul_modular::matrix::Ident;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
/// Writes the contents of a tile to the tensor view using a single plane,
/// iterating with steps determined by the plane's dimension.
pub struct TilewiseUnloader {}

#[cube]
impl PlaneMapper for TilewiseUnloader {
    fn plane_id() -> u32 {
        UNIT_POS_Y
    }

    fn plane_unit() -> u32 {
        UNIT_POS_X
    }
}

#[cube]
impl TilewiseUnloader {
    pub fn unload_from_slice<EG: Numeric, ES: Numeric, G: GmmConfig>(
        write_view: &mut TensorView<EG>,
        slice: &Slice<'_, Line<ES>>,
        tile_x: u32,
        tile_y: u32,
        #[comptime] config: G
    ) {
        let stage_dim = config.stage_dim(Ident::Out);
        let slice_line_size = config.out_smem_line_size();
        let out_line_size = config.line_size(Ident::Out);

        let unit_step = config.plane_dim() * out_line_size;
        let num_unit_writes = stage_dim.tile_num_elements() / unit_step;

        let _ = comptime!(check_line_size(out_line_size, slice_line_size));

        for i in 0..num_unit_writes {
            let unit_write = TilewiseUnloader::plane_unit() * out_line_size + i * unit_step;

            let value = slice[unit_write / out_line_size];
            write_view.write_coalesced::<ES, G>(tile_x, tile_y, unit_write, value, config);
        }
    }
}

fn check_line_size(out_line_size: u32, slice_line_size: u32) {
    assert_eq!(out_line_size, slice_line_size, 
        "Error: Expected global output and output shared memory to have equal line size, but found out_line_size = {} and slice_line_size = {}.",
        out_line_size, slice_line_size
    );
}
