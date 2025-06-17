use crate::components::InputIdent;
use crate::components::global::load::BufferId;
use crate::components::stage::ReaderFamily;
use crate::components::stage::StageConfig;
use crate::components::stage::StageMemory;
use crate::components::stage::StageToTileReader;
use crate::components::stage::TilingLayout;
use crate::components::tile::Tile;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
pub struct FullStageToTileReader<ES: Numeric, T: TilingLayout> {
    pub stage_memory: StageMemory<ES, T>,
    #[cube(comptime)]
    pub input_ident: InputIdent,
}

pub struct FullReaderFamily;

impl ReaderFamily for FullReaderFamily {
    type Reader<ES: Numeric, T: TilingLayout> = FullStageToTileReader<ES, T>;
}

#[cube]
impl<ES: Numeric, T: TilingLayout> FullStageToTileReader<ES, T> {
    pub fn new(stage_memory: StageMemory<ES, T>, #[comptime] input_ident: InputIdent) -> Self {
        FullStageToTileReader::<ES, T> {
            stage_memory,
            input_ident,
        }
    }
}

#[cube]
impl<ES: Numeric, T: TilingLayout> StageToTileReader<ES> for FullStageToTileReader<ES, T> {
    fn read_tile<S: StageConfig>(
        this: &Self,
        row: u32,
        col: u32,
        #[comptime] config: S,
    ) -> Tile<ES> {
        this.stage_memory.get_tile::<S>(
            row,
            col,
            0u32,
            comptime!(this.input_ident.as_ident()),
            config,
        )
    }
}

#[derive(CubeType)]
pub struct BufferStageToTileReader<ES: Numeric, T: TilingLayout> {
    pub stage_memory: StageMemory<ES, T>,
    #[cube(comptime)]
    pub buffer_id: BufferId,
    #[cube(comptime)]
    input_ident: InputIdent,
}

pub struct BufferReaderFamily;

impl ReaderFamily for BufferReaderFamily {
    type Reader<I: Numeric, T: TilingLayout> = BufferStageToTileReader<I, T>;
}

#[cube]
impl<ES: Numeric, T: TilingLayout> BufferStageToTileReader<ES, T> {
    pub fn new(
        stage_memory: StageMemory<ES, T>,
        #[comptime] buffer_id: BufferId,
        #[comptime] input_ident: InputIdent,
    ) -> BufferStageToTileReader<ES, T> {
        BufferStageToTileReader::<ES, T> {
            stage_memory,
            buffer_id,
            input_ident,
        }
    }
}

#[cube]
impl<ES: Numeric, T: TilingLayout> StageToTileReader<ES> for BufferStageToTileReader<ES, T> {
    fn read_tile<S: StageConfig>(
        this: &Self,
        row: u32,
        col: u32,
        #[comptime] config: S,
    ) -> Tile<ES> {
        this.stage_memory.get_tile::<S>(
            row,
            col,
            comptime!(this.buffer_id.to_index()),
            comptime!(this.input_ident.as_ident()),
            config,
        )
    }
}
