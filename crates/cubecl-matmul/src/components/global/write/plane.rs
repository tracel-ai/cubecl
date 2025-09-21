use crate::components::global::memory::GlobalMemoryConfig;
use crate::components::global::memory::TensorWriter;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::View;
use cubecl_std::{div_ceil, tensor::layout::Coords2d};

use super::StageUnloader;

#[derive(CubeType)]
/// Writes tiles from out shared memory to output global memory
/// using a plane for each tile
pub struct PlaneWriter<EG: Numeric> {
    pub tensor_writer: TensorWriter<EG>,
}

#[cube]
impl<EG: Numeric> PlaneWriter<EG> {
    pub fn new(view: View<Line<EG>, Coords2d, ReadWrite>) -> Self {
        PlaneWriter::<EG> {
            tensor_writer: TensorWriter::new(view),
        }
    }
}

#[cube]
impl<EG: Numeric> StageUnloader<EG> for PlaneWriter<EG> {
    type Coordinates = Coords2d;

    fn write(
        this: &mut Self,
        out_smem_slice: Slice<Line<EG>>,
        tile_row: u32,
        tile_col: u32,
        #[comptime] smem_line_size: u32,
        #[comptime] plane_dim: u32,
        #[comptime] config: GlobalMemoryConfig,
    ) {
        let tile_size = config.elements_in_tile_row * config.elements_in_tile_col;
        let output_line_size = config.global_line_size;

        let unit_step = plane_dim * output_line_size;
        let num_unit_writes = comptime!(div_ceil(tile_size, unit_step));
        let balanced_workload = comptime!(tile_size.is_multiple_of(unit_step));

        #[unroll(num_unit_writes == 1)]
        for i in 0..num_unit_writes {
            let unit_write = UNIT_POS_X * output_line_size + i * unit_step;

            #[allow(clippy::collapsible_else_if)]
            if comptime!(balanced_workload) {
                write_line(
                    &mut this.tensor_writer,
                    &out_smem_slice,
                    unit_write,
                    tile_row,
                    tile_col,
                    output_line_size,
                    smem_line_size,
                    config,
                );
            } else {
                if unit_write < tile_size {
                    write_line(
                        &mut this.tensor_writer,
                        &out_smem_slice,
                        unit_write,
                        tile_row,
                        tile_col,
                        output_line_size,
                        smem_line_size,
                        config,
                    );
                }
            }
        }
    }
}

#[cube]
fn write_line<EG: Numeric>(
    tensor_writer: &mut TensorWriter<EG>,
    out_smem_slice: &Slice<Line<EG>>,
    unit_write: u32,
    tile_row: u32,
    tile_col: u32,
    #[comptime] output_line_size: u32,
    #[comptime] out_smem_line_size: u32,
    #[comptime] out_config: GlobalMemoryConfig,
) {
    let value = if comptime!(output_line_size == out_smem_line_size) {
        out_smem_slice[unit_write / output_line_size]
    } else if comptime!(
        out_smem_line_size < output_line_size
            && output_line_size.is_multiple_of(out_smem_line_size)
    ) {
        let mut value = Line::empty(output_line_size);
        #[unroll]
        for i in 0..comptime!(output_line_size / out_smem_line_size) {
            #[unroll]
            for j in 0..out_smem_line_size {
                value[i * out_smem_line_size + j] = out_smem_slice[unit_write + i][j];
            }
        }
        value
    } else {
        unimplemented!()
    };

    tensor_writer.write_coalesced(tile_row, tile_col, unit_write, value, out_config);
}
