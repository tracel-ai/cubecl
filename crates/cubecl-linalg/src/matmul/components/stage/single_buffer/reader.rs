use crate::matmul::components::{
    global::multi_stage::double_buffering::BufferId,
    stage::{DualStage, ReaderFamily, TilingLayout, shared::CommonStageConfig},
    tile::{Tile, TileConfig},
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::Ident;

#[derive(CubeType)]
pub struct LhsBufferReader<ES: Numeric, T: TilingLayout> {
    pub stage: DualStage<ES, T>,
    #[cube(comptime)]
    pub buffer_id: BufferId,
}

#[derive(CubeType)]
pub struct RhsBufferReader<ES: Numeric, T: TilingLayout> {
    pub stage: DualStage<ES, T>,
    #[cube(comptime)]
    pub buffer_id: BufferId,
}

pub struct LhsBufferReaderFamily;
pub struct RhsBufferReaderFamily;

impl ReaderFamily for LhsBufferReaderFamily {
    type Reader<I: Numeric, T: TilingLayout> = LhsBufferReader<I, T>;
}

impl ReaderFamily for RhsBufferReaderFamily {
    type Reader<I: Numeric, T: TilingLayout> = RhsBufferReader<I, T>;
}

#[cube]
impl<ES: Numeric, T: TilingLayout> LhsBufferReader<ES, T> {
    pub fn new(stage: DualStage<ES, T>, #[comptime] buffer_id: BufferId) -> LhsBufferReader<ES, T> {
        LhsBufferReader::<ES, T> { stage, buffer_id }
    }

    pub fn read_tile<TC: TileConfig>(
        this: &Self,
        compute_plane_offset: u32,
        k_offset: u32,
        #[comptime] config: CommonStageConfig<TC>,
    ) -> Tile<ES> {
        this.stage.get_tile::<CommonStageConfig<TC>>(
            compute_plane_offset,
            k_offset,
            this.buffer_id,
            Ident::Lhs,
            config,
        )
    }
}

#[cube]
impl<ES: Numeric, T: TilingLayout> RhsBufferReader<ES, T> {
    pub fn new(stage: DualStage<ES, T>, #[comptime] buffer_id: BufferId) -> RhsBufferReader<ES, T> {
        RhsBufferReader::<ES, T> { stage, buffer_id }
    }

    pub fn read_tile<TC: TileConfig>(
        this: &Self,
        accumulator_offset: u32,
        k_offset: u32,
        #[comptime] config: CommonStageConfig<TC>,
    ) -> Tile<ES> {
        this.stage.get_tile::<CommonStageConfig<TC>>(
            accumulator_offset,
            k_offset,
            this.buffer_id,
            Ident::Rhs,
            config,
        )
    }
}
