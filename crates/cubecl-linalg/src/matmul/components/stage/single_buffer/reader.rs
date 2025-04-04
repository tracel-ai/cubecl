use crate::matmul::components::{
    InputIdent,
    global::multi_stage::double_buffering::BufferId,
    stage::{ReaderFamily, TilingLayout, shared::CommonStageConfig},
    tile::{Tile, TileConfig},
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::stage::Stage;

#[derive(CubeType)]
pub struct BufferReader<ES: Numeric, T: TilingLayout> {
    pub stage: Stage<ES, T>,
    #[cube(comptime)]
    pub buffer_id: BufferId,
    #[cube(comptime)]
    input_ident: InputIdent,
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
        #[comptime] input_ident: InputIdent,
    ) -> BufferReader<ES, T> {
        BufferReader::<ES, T> {
            stage,
            buffer_id,
            input_ident,
        }
    }

    pub fn read_tile<TC: TileConfig>(
        this: &Self,
        compute_plane_offset: u32,
        #[comptime] config: CommonStageConfig<TC>,
    ) -> Tile<ES> {
        let buffer_index = comptime!(this.buffer_id.to_index());
        let (x, y) = match comptime!(this.input_ident) {
            InputIdent::Lhs => (compute_plane_offset, buffer_index),
            InputIdent::Rhs => (buffer_index, compute_plane_offset),
        };
        this.stage.get_tile::<CommonStageConfig<TC>>(
            x,
            y,
            comptime!(this.input_ident.as_ident()),
            config,
        )
    }
}
