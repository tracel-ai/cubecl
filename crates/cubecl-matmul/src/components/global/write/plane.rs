use crate::components::{
    global::{
        memory::GlobalMemoryConfig,
        read::tiled::{TiledCoords, TiledLayout},
    },
    tile::{StridedTile, io::Strided},
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::View;
use cubecl_std::tensor::layout::Coords2d;

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
    type TileKind = Strided;

    fn write<ES: Numeric>(
        this: &mut Self,
        smem_tile: &StridedTile<ES, ReadWrite>,
        tile: Coords2d,
        #[comptime] plane_dim: u32,
        #[comptime] config: GlobalMemoryConfig,
    ) {
        let tile_size = config.elements_in_tile_row * config.elements_in_tile_col;
        let output_line_size = config.global_line_size;

        let unit_step = plane_dim * output_line_size;
        let num_unit_writes = comptime!(tile_size.div_ceil(unit_step));
        let balanced_workload = comptime!(tile_size.is_multiple_of(unit_step));

        #[unroll(num_unit_writes == 1)]
        for i in 0..num_unit_writes {
            let unit_write = UNIT_POS_X * output_line_size + i * unit_step;

            #[allow(clippy::collapsible_else_if)]
            if comptime!(balanced_workload) {
                write_line(&mut this.view, &smem_tile.slice, unit_write, tile);
            } else {
                if unit_write < tile_size {
                    write_line(&mut this.view, &smem_tile.slice, unit_write, tile);
                }
            }
        }
    }
}

#[cube]
fn write_line<ES: Numeric, EG: Numeric>(
    view: &mut View<Line<EG>, TiledCoords, ReadWrite>,
    out_smem_slice: &Slice<Line<ES>, ReadWrite>,
    unit_write: u32,
    tile: Coords2d,
) {
    let output_line_size = view.line_size();
    let out_smem_line_size = out_smem_slice.line_size();

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

    view.write_checked((tile, unit_write), Line::cast_from(value));
}
