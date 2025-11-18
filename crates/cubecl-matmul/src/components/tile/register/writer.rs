use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::components::tile::{
    StridedTile,
    register::{UNROLL, UnitFragment, config::RegisterMatmulConfig},
};

/// Writer for the register matmul fragments.
#[derive(CubeType)]
pub struct RegisterStageWriter {}

#[cube]
impl RegisterStageWriter {
    pub fn store_fragment<A: Numeric, E: Numeric>(
        tile: &mut StridedTile<E, ReadWrite>,
        acc: &UnitFragment<A>,
        #[comptime] config: RegisterMatmulConfig,
    ) {
        let out_line_size = tile.stage.line_size();

        #[unroll(UNROLL)]
        for i in 0..comptime!(config.shared.tile_size.mn() / out_line_size) {
            let offs = tile.stage_offset(i);
            let mut line = Line::empty(out_line_size);
            #[unroll]
            for j in 0..comptime!(out_line_size) {
                line[j] = acc.array[i * out_line_size + j];
            }
            tile.stage[offs] = Line::cast_from(line);
        }
    }
}
