use crate::matmul::components::{
    global::multi_stage::double_buffering::BufferId,
    stage::{ReaderFamily, TilingLayout, shared::CommonStageConfig},
    tile::{Tile, TileConfig},
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::{Ident, stage::Stage};

#[derive(CubeType)]
pub struct BufferReader<ES: Numeric, T: TilingLayout> {
    pub stage: Stage<ES, T>,
    #[cube(comptime)]
    pub buffer_id: BufferId,
    #[cube(comptime)]
    ident: Ident,
}

pub struct BufferReaderFamily;

impl ReaderFamily for BufferReaderFamily {
    type Reader<I: Numeric, T: TilingLayout> = BufferReader<I, T>;
}

#[cube]
impl<ES: Numeric, T: TilingLayout> BufferReader<ES, T> {
    pub fn new(
        stage: Stage<ES, T>,
        #[comptime] buffer_id: BufferId,
        #[comptime] ident: Ident,
    ) -> BufferReader<ES, T> {
        BufferReader::<ES, T> {
            stage,
            buffer_id,
            ident,
        }
    }

    pub fn read_tile<TC: TileConfig>(
        this: &Self,
        compute_plane_offset: u32,
        #[comptime] config: CommonStageConfig<TC>,
    ) -> Tile<ES> {
        this.stage.get_tile::<CommonStageConfig<TC>>(
            compute_plane_offset,
            comptime!(this.buffer_id.to_u32()),
            Ident::Lhs,
            config,
        )
    }
}
