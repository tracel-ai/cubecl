use crate::matmul::components::global::tensor_view::TensorWriter;
use crate::matmul::components::global::GlobalConfig;
use crate::matmul::components::Ident;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
/// Writes the contents of a tile to the tensor view using a single plane,
/// iterating with steps determined by the plane's dimension.
pub struct TilewiseUnloading {}

#[cube]
impl TilewiseUnloading {
    pub fn unload_from_slice<EG: Numeric, ES: Numeric, G: GlobalConfig>(
        write_view: &mut TensorWriter<EG>,
        slice: Slice<Line<ES>>,
        tile_x: u32,
        tile_y: u32,
        #[comptime] config: G,
    ) {
        let tiling = config.tiling_dimensions(Ident::Out);
        let slice_line_size = config.stage_line_size(Ident::Out);
        let out_line_size = config.global_line_size(Ident::Out);

        let unit_step = config.plane_dim() * out_line_size;
        let num_unit_writes = tiling.tile_size() / unit_step;

        #[allow(clippy::all)]
        let _ = comptime!(check_line_size(out_line_size, slice_line_size));

        for i in 0..num_unit_writes {
            let unit_write = UNIT_POS_X * out_line_size + i * unit_step;

            let value = slice[unit_write / out_line_size];
            write_view.write_coalesced::<ES, G>(tile_x, tile_y, unit_write, value, config);
        }
    }
}

fn check_line_size(out_line_size: u32, slice_line_size: u32) {
    assert_eq!(out_line_size, slice_line_size, "Error: Expected global output and output shared memory to have equal line size, but found out_line_size = {} and slice_line_size = {}.",
        out_line_size, slice_line_size
    );
}
