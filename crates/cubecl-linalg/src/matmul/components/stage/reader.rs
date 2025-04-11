use crate::matmul::components::Ident;
use crate::matmul::components::InputIdent;
use crate::matmul::components::global::load::BufferId;
use crate::matmul::components::stage::ReaderFamily;
use crate::matmul::components::stage::Stage;
use crate::matmul::components::stage::StageConfig;
use crate::matmul::components::stage::TilingLayout;
use crate::matmul::components::stage::shared::CommonStageConfig;
use crate::matmul::components::tile::Tile;
use crate::matmul::components::tile::TileConfig;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub trait Reader<ES: Numeric>: CubeType + Send + Sync + 'static {
    fn num_k_iterations<TC: TileConfig>(
        #[comptime] config: CommonStageConfig<TC>,
    ) -> comptime_type!(u32);

    fn read_tile<TC: TileConfig>(
        this: &Self,
        row: u32,
        col: u32,
        #[comptime] config: CommonStageConfig<TC>,
    ) -> Tile<ES>;
}

#[derive(CubeType)]
pub struct FullReader<ES: Numeric, T: TilingLayout> {
    pub stage: Stage<ES, T>,
    #[cube(comptime)]
    pub input_ident: InputIdent,
}

pub struct FullReaderFamily;

impl ReaderFamily for FullReaderFamily {
    type Reader<ES: Numeric, T: TilingLayout> = FullReader<ES, T>;
}

#[cube]
impl<ES: Numeric, T: TilingLayout> FullReader<ES, T> {
    pub fn new(stage: Stage<ES, T>, #[comptime] input_ident: InputIdent) -> Self {
        FullReader::<ES, T> { stage, input_ident }
    }
}

#[cube]
impl<ES: Numeric, T: TilingLayout> Reader<ES> for FullReader<ES, T> {
    fn num_k_iterations<TC: TileConfig>(
        #[comptime] config: CommonStageConfig<TC>,
    ) -> comptime_type!(u32) {
        config.tiling_dimensions(Ident::Lhs).tile_count_col()
    }

    fn read_tile<TC: TileConfig>(
        this: &Self,
        row: u32,
        col: u32,
        #[comptime] config: CommonStageConfig<TC>,
    ) -> Tile<ES> {
        this.stage.get_tile::<CommonStageConfig<TC>>(
            row,
            col,
            comptime!(this.input_ident.as_ident()),
            config,
        )
    }
}

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
}

#[cube]
impl<ES: Numeric, T: TilingLayout> Reader<ES> for BufferReader<ES, T> {
    fn num_k_iterations<TC: TileConfig>(
        #[comptime] _config: CommonStageConfig<TC>,
    ) -> comptime_type!(u32) {
        // For now buffers are always assumed to have 1 k tile
        1u32
    }

    fn read_tile<TC: TileConfig>(
        this: &Self,
        row: u32,
        col: u32,
        #[comptime] config: CommonStageConfig<TC>,
    ) -> Tile<ES> {
        let buffer_index = comptime!(this.buffer_id.to_index());
        let (x, y) = match comptime!(this.input_ident) {
            InputIdent::Lhs => (row, buffer_index),
            InputIdent::Rhs => (buffer_index, col),
        };
        this.stage.get_tile::<CommonStageConfig<TC>>(
            x,
            y,
            comptime!(this.input_ident.as_ident()),
            config,
        )
    }
}
