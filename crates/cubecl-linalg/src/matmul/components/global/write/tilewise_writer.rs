use crate::matmul::components::Ident;
use crate::matmul::components::global::GlobalConfig;
use crate::matmul::components::global::tensor_view::TensorWriter;
use crate::matmul::components::stage::Writer;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::div_ceil;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};

#[derive(CubeType)]
pub struct TilewiseWriter<EG: Numeric> {
    pub tensor_view: TensorWriter<EG>,
}

#[cube]
impl<EG: Numeric> TilewiseWriter<EG> {
    pub fn new(
        tensor: VirtualTensor<EG, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
    ) -> Self {
        TilewiseWriter::<EG> {
            tensor_view: TensorWriter::new(tensor, x_offset, y_offset, batch_offset),
        }
    }
}

#[cube]
impl<EG: Numeric> Writer<EG> for TilewiseWriter<EG> {
    fn write<ES: Numeric, G: GlobalConfig>(
        this: &mut Self,
        out_smem_slice: Slice<Line<ES>>,
        tile_row: u32,
        tile_col: u32,
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
                this.tensor_view
                    .write_coalesced::<ES, G>(tile_row, tile_col, unit_write, value, config);
            } else {
                if unit_write < tile_size {
                    let value = out_smem_slice[unit_write / output_line_size];
                    this.tensor_view
                        .write_coalesced::<ES, G>(tile_row, tile_col, unit_write, value, config);
                }
            }
        }
    }
}
