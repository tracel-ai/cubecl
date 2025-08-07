use crate::components::MatmulIdent;
use crate::components::global::GlobalConfig;
use crate::components::global::memory::TensorWriter;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::div_ceil;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};

use super::GlobalWriter;

#[derive(CubeType)]
/// Writes tiles from out shared memory to output global memory
/// using a plane for each tile
pub struct PlaneWriter<EG: Numeric> {
    pub tensor_writer: TensorWriter<EG>,
}

#[cube]
impl<EG: Numeric> PlaneWriter<EG> {
    pub fn new(
        tensor: VirtualTensor<EG, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
    ) -> Self {
        PlaneWriter::<EG> {
            tensor_writer: TensorWriter::new(tensor, x_offset, y_offset, batch_offset),
        }
    }
}

#[cube]
impl<EG: Numeric> GlobalWriter<EG> for PlaneWriter<EG> {
    fn write<G: GlobalConfig>(
        this: &mut Self,
        out_smem_slice: Slice<Line<EG>>,
        tile_row: u32,
        tile_col: u32,
        #[comptime] config: G,
    ) {
        let tile_size = config.tiling_scheme().elements_in_tile_mn();
        let output_line_size = config.global_line_size(MatmulIdent::Out);
        let out_config = config.global_memory_config(MatmulIdent::Out);

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
