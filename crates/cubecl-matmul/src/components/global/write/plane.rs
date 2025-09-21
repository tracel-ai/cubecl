use crate::components::global::{
    GlobalConfig,
    read::tiled::{TiledCoords, TiledLayout},
};
use crate::components::{
    MatmulIdent, StageIdent, global::memory::GlobalMemoryConfig, stage::StageConfig,
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::View;
use cubecl_std::{div_ceil, tensor::layout::Coords2d};

use super::GlobalWriter;

#[derive(CubeType)]
/// Writes tiles from out shared memory to output global memory
/// using a plane for each tile
pub struct PlaneWriter<EG: Numeric> {
    pub view: View<Line<EG>, TiledCoords, ReadWrite>,
}

#[cube]
impl<EG: Numeric> PlaneWriter<EG> {
    pub fn new(
        view: View<Line<EG>, Coords2d, ReadWrite>,
        #[comptime] config: GlobalMemoryConfig,
    ) -> Self {
        PlaneWriter::<EG> {
            view: view.view_mut(TiledLayout::new(config)),
        }
    }
}

#[cube]
impl<EG: Numeric> GlobalWriter<EG> for PlaneWriter<EG> {
    type Coordinates = Coords2d;

    fn write<G: GlobalConfig>(
        this: &mut Self,
        out_smem_slice: Slice<Line<EG>>,
        tile: Coords2d,
        #[comptime] config: G,
    ) {
        let tile_size = config.tiling_scheme().elements_in_tile_mn();
        let output_line_size = config.global_line_size(MatmulIdent::Out);

        let out_smem_line_size = config.stage_config().stage_line_size(StageIdent::Acc);

        let unit_step = config.plane_dim() * output_line_size;
        let num_unit_writes = comptime!(div_ceil(tile_size, unit_step));
        let balanced_workload = comptime!(tile_size.is_multiple_of(unit_step));

        #[unroll(num_unit_writes == 1)]
        for i in 0..num_unit_writes {
            let unit_write = UNIT_POS_X * output_line_size + i * unit_step;

            #[allow(clippy::collapsible_else_if)]
            if comptime!(balanced_workload) {
                write_line(
                    &mut this.view,
                    &out_smem_slice,
                    unit_write,
                    tile,
                    output_line_size,
                    out_smem_line_size,
                );
            } else {
                if unit_write < tile_size {
                    write_line(
                        &mut this.view,
                        &out_smem_slice,
                        unit_write,
                        tile,
                        output_line_size,
                        out_smem_line_size,
                    );
                }
            }
        }
    }
}

#[cube]
fn write_line<EG: Numeric>(
    view: &mut View<Line<EG>, TiledCoords, ReadWrite>,
    out_smem_slice: &Slice<Line<EG>>,
    unit_write: u32,
    tile: Coords2d,
    #[comptime] output_line_size: u32,
    #[comptime] out_smem_line_size: u32,
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

    view.write_checked((tile, unit_write), value);
}
