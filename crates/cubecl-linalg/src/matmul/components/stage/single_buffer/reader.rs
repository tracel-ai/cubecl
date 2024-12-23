use crate::matmul::components::{
    stage::{shared::CommonStageConfig, ReaderFamily},
    tile::TileConfig,
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::{stage::Stage, Ident};

#[derive(CubeType)]
pub struct LhsBufferReader<ES: Numeric> {
    pub stage: Stage<ES>,
    pub buffer: u32,
}

#[derive(CubeType)]
pub struct RhsBufferReader<ES: Numeric> {
    pub stage: Stage<ES>,
    pub buffer: u32,
}

pub struct LhsBufferReaderFamily;
pub struct RhsBufferReaderFamily;

impl ReaderFamily for LhsBufferReaderFamily {
    type Reader<I: Numeric> = LhsBufferReader<I>;
}

impl ReaderFamily for RhsBufferReaderFamily {
    type Reader<I: Numeric> = RhsBufferReader<I>;
}

#[cube]
impl<ES: Numeric> LhsBufferReader<ES> {
    pub fn read_tile<T: TileConfig>(
        this: &Self,
        compute_plane_offset: u32,
        #[comptime] config: CommonStageConfig<T>,
    ) -> Slice<Line<ES>> {
        this.stage.get_tile::<CommonStageConfig<T>>(
            compute_plane_offset,
            this.buffer,
            Ident::Lhs,
            config,
        )
    }
}

#[cube]
impl<ES: Numeric> RhsBufferReader<ES> {
    pub fn read_tile<T: TileConfig>(
        this: &Self,
        accumulator_offset: u32,
        #[comptime] config: CommonStageConfig<T>,
    ) -> Slice<Line<ES>> {
        this.stage.get_tile::<CommonStageConfig<T>>(
            this.buffer,
            accumulator_offset,
            Ident::Rhs,
            config,
        )
    }
}
