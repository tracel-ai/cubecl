use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::{global::memory::TensorWriter, stage::StageMemoryConfig as _};
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};

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
        tensor: VirtualTensor<EO, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
    ) -> Self {
        DummyWriter::<EO> {
            tensor_writer: TensorWriter::new(tensor, x_offset, y_offset, batch_offset),
        }
    }

    fn write<G: GlobalAttentionConfig>(
        this: &mut Self,
        out_smem_slice: Slice<Line<EO>>,
        tile_row: u32,
        tile_col: u32,
        #[comptime] config: G,
    ) {
        // let tile_size = config
        //     .value_stage_memory_config()
        //     .tiling_scheme()
        //     .elements_in_tile_mn();
        // let output_line_size = config.global_line_size(FlashIdent::Out);
        // let out_smem_slice = out_smem_slice.with_line_size(output_line_size);

        // let num_lines = tile_size / output_line_size;

        // for i in 0..num_lines {
        //     let value = out_smem_slice[i];
        //     this.tensor_writer.write_coalesced::<G>(
        //         tile_row,
        //         tile_col,
        //         i * output_line_size,
        //         value,
        //         config,
        //     );
        // }
    }
}
