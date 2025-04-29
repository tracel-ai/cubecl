use crate::matmul::components::InputIdent;
use crate::matmul::components::global::load::BufferId;
use crate::matmul::components::stage::ReaderFamily;
use crate::matmul::components::stage::StageMemory;
use crate::matmul::components::stage::TilingLayout;
use crate::matmul::components::stage::shared::CommonStageConfig;
use crate::matmul::components::tile::Tile;
use crate::matmul::components::tile::TileConfig;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub trait Reader<ES: Numeric>: CubeType + Send + Sync + 'static {
    fn read_tile<TC: TileConfig>(
        this: &Self,
        row: u32,
        col: u32,
        #[comptime] config: CommonStageConfig<TC>,
    ) -> Tile<ES>;
}

#[derive(CubeType)]
pub struct FullReader<ES: Numeric, T: TilingLayout> {
    pub stage_memory: StageMemory<ES, T>,
    #[cube(comptime)]
    pub input_ident: InputIdent,
}

pub struct FullReaderFamily;

impl ReaderFamily for FullReaderFamily {
    type Reader<ES: Numeric, T: TilingLayout> = FullReader<ES, T>;
}

#[cube]
impl<ES: Numeric, T: TilingLayout> FullReader<ES, T> {
    pub fn new(stage_memory: StageMemory<ES, T>, #[comptime] input_ident: InputIdent) -> Self {
        FullReader::<ES, T> {
            stage_memory,
            input_ident,
        }
    }
}

#[cube]
impl<ES: Numeric, T: TilingLayout> Reader<ES> for FullReader<ES, T> {
    fn read_tile<TC: TileConfig>(
        this: &Self,
        row: u32,
        col: u32,
        #[comptime] config: CommonStageConfig<TC>,
    ) -> Tile<ES> {
        this.stage_memory.get_tile::<CommonStageConfig<TC>>(
            row,
            col,
            0u32,
            comptime!(this.input_ident.as_ident()),
            config,
        )
    }
}

#[derive(CubeType)]
pub struct BufferReader<ES: Numeric, T: TilingLayout> {
    pub stage_memory: StageMemory<ES, T>,
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
        stage_memory: StageMemory<ES, T>,
        #[comptime] buffer_id: BufferId,
        #[comptime] input_ident: InputIdent,
    ) -> BufferReader<ES, T> {
        BufferReader::<ES, T> {
            stage_memory,
            buffer_id,
            input_ident,
        }
    }
}

#[cube]
impl<ES: Numeric, T: TilingLayout> Reader<ES> for BufferReader<ES, T> {
    fn read_tile<TC: TileConfig>(
        this: &Self,
        row: u32,
        col: u32,
        #[comptime] config: CommonStageConfig<TC>,
    ) -> Tile<ES> {
        this.stage_memory.get_tile::<CommonStageConfig<TC>>(
            row,
            col,
            comptime!(this.buffer_id.to_index()),
            comptime!(this.input_ident.as_ident()),
            config,
        )
    }
}
