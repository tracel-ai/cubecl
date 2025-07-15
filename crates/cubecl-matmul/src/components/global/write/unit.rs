use crate::components::Ident;
use crate::components::global::GlobalConfig;
use crate::components::global::global_memory::TensorWriter;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};

use super::GlobalWriter;

#[derive(CubeType)]
/// Writes tiles from out shared memory to output global memory
/// using a unit for each tile
pub struct UnitWriter<EG: Numeric> {
    pub tensor_view: TensorWriter<EG>,
}

#[cube]
impl<EG: Numeric> UnitWriter<EG> {
    pub fn new(
        tensor: VirtualTensor<EG, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
    ) -> Self {
        UnitWriter::<EG> {
            tensor_view: TensorWriter::new(tensor, x_offset, y_offset, batch_offset),
        }
    }
}

#[cube]
impl<EG: Numeric> GlobalWriter<EG> for UnitWriter<EG> {
    fn write<G: GlobalConfig>(
        this: &mut Self,
        out_smem_slice: Slice<Line<EG>>,
        tile_row: u32,
        tile_col: u32,
        #[comptime] config: G,
    ) {
        let tile_size = config.tiling_scheme().elements_in_tile_mn();
        let output_line_size = config.global_line_size(Ident::Out);
        let out_smem_slice = out_smem_slice.with_line_size(output_line_size);

        let num_lines = tile_size / output_line_size;

        for i in 0..num_lines {
            let value = out_smem_slice[i];
            this.tensor_view.write_coalesced::<G>(
                tile_row,
                tile_col,
                i * output_line_size,
                value,
                config,
            );
        }
    }
}
