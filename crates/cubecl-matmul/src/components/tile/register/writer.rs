use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::components::tile::{
    StridedTile,
    register::{UNROLL, config::RegisterConfig},
};

/// Writer for the register matmul fragments.
#[derive(CubeType)]
pub struct RegisterStageWriter {}

#[cube]
impl RegisterStageWriter {
    pub fn store_fragment<A: Numeric, E: Numeric>(
        tile: &mut StridedTile<E, ReadWrite>,
        acc: &Array<A>,
        #[comptime] config: RegisterConfig,
    ) {
        let out_line_size = tile.slice.line_size();
        #[unroll(UNROLL)]
        for i in 0..comptime!(config.tile_size.mn() / out_line_size) {
            let mut line = Line::empty(out_line_size);
            #[unroll(UNROLL)]
            for j in 0..comptime!(out_line_size) {
                line[j] = acc[i * out_line_size + j];
            }
            tile.slice[i] = Line::cast_from(line);
        }
    }
}
