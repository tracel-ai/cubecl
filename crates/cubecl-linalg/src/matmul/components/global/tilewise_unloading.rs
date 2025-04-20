use crate::matmul::components::Ident;
use crate::matmul::components::global::GlobalConfig;
use crate::matmul::components::global::tensor_view::TensorWriter;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::div_ceil;

#[derive(CubeType)]
/// Writes the contents of a tile to the tensor view using a single plane,
/// iterating with steps determined by the plane's dimension.
pub struct TilewiseUnloading {}

#[cube]
impl TilewiseUnloading {
    pub fn unload_from_slice<EG: Numeric, ES: Numeric, G: GlobalConfig>(
        write_view: &mut TensorWriter<EG>,
        out_smem_slice: Slice<Line<ES>>,
        tile_x: u32,
        tile_y: u32,
        #[comptime] config: G,
    ) {
        let tiling = config.tiling_dimensions(Ident::Out);
        let tile_size = tiling.tile_size();
        let output_line_size = config.global_line_size(Ident::Out);
        let out_smem_slice = out_smem_slice.with_line_size(output_line_size);

        let unit_step = config.plane_dim() * output_line_size;
        let num_unit_writes = comptime!(div_ceil(tile_size, unit_step));
        let balanced_workload = comptime!(tile_size % unit_step == 0);

        #[unroll(num_unit_writes == 1)]
        for i in 0..num_unit_writes {
            let unit_write = UNIT_POS_X * output_line_size + i * unit_step;

            #[allow(clippy::collapsible_else_if)]
            if comptime!(balanced_workload) {
                let value = out_smem_slice[unit_write / output_line_size];
                write_view.write_coalesced::<ES, G>(tile_x, tile_y, unit_write, value, config);
            } else {
                if unit_write < tile_size {
                    let value = out_smem_slice[unit_write / output_line_size];
                    write_view.write_coalesced::<ES, G>(tile_x, tile_y, unit_write, value, config);
                }
            }
        }
    }
}
