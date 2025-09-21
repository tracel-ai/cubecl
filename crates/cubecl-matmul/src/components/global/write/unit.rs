use crate::components::global::memory::{GlobalMemoryConfig, TensorWriter};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::{View, layout::Coords2d};

use super::StageUnloader;

#[derive(CubeType)]
/// Writes tiles from out shared memory to output global memory
/// using a unit for each tile
pub struct UnitWriter<EG: Numeric> {
    pub tensor_view: TensorWriter<EG>,
}

#[cube]
impl<EG: Numeric> UnitWriter<EG> {
    pub fn new(view: View<Line<EG>, Coords2d, ReadWrite>) -> Self {
        UnitWriter::<EG> {
            tensor_view: TensorWriter::new(view),
        }
    }
}

#[cube]
impl<EG: Numeric> StageUnloader<EG> for UnitWriter<EG> {
    type Coordinates = Coords2d;

    fn write(
        this: &mut Self,
        out_smem_slice: Slice<Line<EG>>,
        tile_row: u32,
        tile_col: u32,
        #[comptime] _smem_line_size: u32,
        #[comptime] _plane_dim: u32,
        #[comptime] config: GlobalMemoryConfig,
    ) {
        let tile_size = config.elements_in_tile_row * config.elements_in_tile_col;
        let output_line_size = config.global_line_size;
        let out_smem_slice = out_smem_slice.with_line_size(output_line_size);

        let num_lines = tile_size / output_line_size;

        for i in 0..num_lines {
            let value = out_smem_slice[i];
            this.tensor_view.write_coalesced(
                tile_row,
                tile_col,
                i * output_line_size,
                value,
                config,
            );
        }
    }
}
