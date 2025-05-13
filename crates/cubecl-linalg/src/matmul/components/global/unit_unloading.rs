use crate::matmul::components::Ident;
use crate::matmul::components::global::GlobalConfig;
use crate::matmul::components::global::tensor_view::TensorWriter;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
/// Writes the contents of a tile to the tensor view using a single plane,
/// iterating with steps determined by the plane's dimension.
pub struct UnitUnloading {}

#[cube]
impl UnitUnloading {
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

        let num_lines = tile_size / output_line_size;

        for i in 0..num_lines {
            let value = out_smem_slice[i];
            write_view.write_coalesced::<ES, G>(
                tile_x,
                tile_y,
                i * output_line_size,
                value,
                config,
            );
        }
    }
}
