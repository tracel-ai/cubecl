use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::{global::memory::TensorWriter, stage::StageMemoryConfig as _};
use cubecl_std::{
    div_ceil,
    tensor::{View, layout::Coords3d, r#virtual::ReadWrite},
};

use crate::components::{FlashIdent, global::GlobalAttentionConfig};

#[derive(CubeType)]
/// Writes tiles from out shared memory to output global memory
/// using a unit for each tile
pub struct DummyWriter<EO: Numeric> {
    pub tensor_writer: TensorWriter<EO>,
}

#[cube]
impl<EO: Numeric> DummyWriter<EO> {
    pub fn new(
        tensor: View<EO, Coords3d, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
    ) -> Self {
        DummyWriter::<EO> {
            tensor_writer: TensorWriter::new(tensor, x_offset, y_offset, batch_offset),
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
        let out_config = config.global_memory_config(FlashIdent::Out);

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
                this.tensor_writer
                    .write_coalesced(tile_row, tile_col, unit_write, value, out_config);
            } else {
                if unit_write < tile_size {
                    let value = out_smem_slice[unit_write / output_line_size];
                    this.tensor_writer
                        .write_coalesced(tile_row, tile_col, unit_write, value, out_config);
                }
            }
        }
    }
}
