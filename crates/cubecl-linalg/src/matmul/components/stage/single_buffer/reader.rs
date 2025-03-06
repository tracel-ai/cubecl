use crate::matmul::components::{
    stage::{shared::CommonStageConfig, ReaderFamily, TilingLayout},
    tile::{Tile, TileConfig},
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::{stage::Stage, Ident};

#[derive(CubeType)]
pub struct LhsBufferReader<ES: Numeric, T: TilingLayout> {
    pub stage: Stage<ES, T>,
    pub buffer: u32,
}

#[derive(CubeType)]
pub struct RhsBufferReader<ES: Numeric, T: TilingLayout> {
    pub stage: Stage<ES, T>,
    pub buffer: u32,
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
    pub fn read_tile<TC: TileConfig>(
        this: &Self,
        compute_plane_offset: u32,
        #[comptime] config: CommonStageConfig<TC>,
    ) -> Tile<ES> {
        this.stage.get_tile::<CommonStageConfig<TC>>(
            compute_plane_offset,
            this.buffer,
            Ident::Lhs,
            config,
        )
    }
}

#[cube]
impl<ES: Numeric, T: TilingLayout> RhsBufferReader<ES, T> {
    pub fn read_tile<TC: TileConfig>(
        this: &Self,
        accumulator_offset: u32,
        #[comptime] config: CommonStageConfig<TC>,
    ) -> Tile<ES> {
        this.stage.get_tile::<CommonStageConfig<TC>>(
            this.buffer,
            accumulator_offset,
            Ident::Rhs,
            config,
        )
    }
}
