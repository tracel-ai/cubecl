use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::{
    global::{
        load::tiled::{TiledCoords, TiledLayout},
        memory::GlobalMemoryConfig,
    },
    stage::StageMemoryConfig as _,
};
use cubecl_std::{
    div_ceil,
    tensor::{View, layout::Coords2d},
};

use crate::components::global::GlobalAttentionConfig;

#[derive(CubeType)]
/// Writes tiles from out shared memory to output global memory
/// using a unit for each tile
pub struct DummyWriter<EO: Numeric> {
    pub view: View<Line<EO>, TiledCoords, ReadWrite>,
}

#[cube]
impl<EO: Numeric> DummyWriter<EO> {
    pub fn new(
        tensor: View<Line<EO>, Coords2d, ReadWrite>,
        #[comptime] config: GlobalMemoryConfig,
    ) -> Self {
        DummyWriter::<EO> {
            view: tensor.view_mut(TiledLayout::new(config)),
        }
    }

    pub fn write<G: GlobalAttentionConfig>(
        this: &mut Self,
        out_smem_slice: Slice<Line<EO>>,
        tile_row: u32,
        tile_col: u32,
        #[comptime] config: G,
    ) {
        let tile_size = config
            .value_stage_memory_config()
            .tiling_scheme()
            .elements_in_tile_mn();

        // let output_line_size = config.global_line_size(MatmulIdent::Out);
        let output_line_size = 1;

        let out_smem_slice = out_smem_slice.with_line_size(output_line_size);

        let unit_step = config.plane_dim() * output_line_size;
        let num_unit_writes = comptime!(div_ceil(tile_size, unit_step));
        let balanced_workload = comptime!(tile_size.is_multiple_of(unit_step));

        #[unroll(num_unit_writes == 1)]
        for i in 0..num_unit_writes {
            let unit_write = UNIT_POS_X * output_line_size + i * unit_step;

            #[allow(clippy::collapsible_else_if)]
            if comptime!(balanced_workload) {
                let value = out_smem_slice[unit_write / output_line_size];
                this.view
                    .write_checked(((tile_row, tile_col), unit_write), value);
            } else {
                if unit_write < tile_size {
                    let value = out_smem_slice[unit_write / output_line_size];
                    this.view
                        .write_checked(((tile_row, tile_col), unit_write), value);
                }
            }
        }
    }
}
