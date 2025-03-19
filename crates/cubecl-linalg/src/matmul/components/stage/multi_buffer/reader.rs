use crate::matmul::components::Ident;
use crate::matmul::components::stage::ReaderFamily;
use crate::matmul::components::stage::Stage;
use crate::matmul::components::stage::TilingLayout;
use crate::matmul::components::stage::shared::CommonStageConfig;
use crate::matmul::components::tile::Tile;
use crate::matmul::components::tile::TileConfig;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
/// Stage reader for LHS
pub struct LhsReader<ES: Numeric, T: TilingLayout> {
    pub stage: Stage<ES, T>,
}

#[derive(CubeType)]
/// Stage reader for RHS
pub struct RhsReader<ES: Numeric, T: TilingLayout> {
    pub stage: Stage<ES, T>,
}

pub struct LhsReaderFamily;
pub struct RhsReaderFamily;

impl ReaderFamily for LhsReaderFamily {
    type Reader<I: Numeric, T: TilingLayout> = LhsReader<I, T>;
}

impl ReaderFamily for RhsReaderFamily {
    type Reader<I: Numeric, T: TilingLayout> = RhsReader<I, T>;
}

#[cube]
impl<ES: Numeric, T: TilingLayout> LhsReader<ES, T> {
    pub fn read_tile<TC: TileConfig>(
        this: &Self,
        compute_plane_offset: u32,
        buffer_offset: u32,
        #[comptime] config: CommonStageConfig<TC>,
    ) -> Tile<ES> {
        this.stage.get_tile::<CommonStageConfig<TC>>(
            compute_plane_offset,
            buffer_offset,
            Ident::Lhs,
            config,
        )
    }
}

#[cube]
impl<ES: Numeric, T: TilingLayout> RhsReader<ES, T> {
    pub fn read_tile<TC: TileConfig>(
        this: &Self,
        buffer_offset: u32,
        accumulator_offset: u32,
        #[comptime] config: CommonStageConfig<TC>,
    ) -> Tile<ES> {
        this.stage.get_tile::<CommonStageConfig<TC>>(
            buffer_offset,
            accumulator_offset,
            Ident::Rhs,
            config,
        )
    }
}

#[cube]
impl<ES: Numeric, T: TilingLayout> LhsReader<ES, T> {
    pub fn new(stage: Stage<ES, T>) -> LhsReader<ES, T> {
        LhsReader::<ES, T> { stage }
    }
}

#[cube]
impl<ES: Numeric, T: TilingLayout> RhsReader<ES, T> {
    pub fn new(stage: Stage<ES, T>) -> RhsReader<ES, T> {
        RhsReader::<ES, T> { stage }
    }
}
