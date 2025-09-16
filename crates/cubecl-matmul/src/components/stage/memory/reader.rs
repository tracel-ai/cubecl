use crate::components::stage::StageMemoryConfig;
use crate::components::stage::StageReaderFamily;
use crate::components::stage::TilingLayout;
use crate::components::tile::Tile;
use crate::components::{StageIdent, stage::StageReader};
use crate::components::{global::load::StageBuffer, tile::loader::Strided};
use crate::components::{stage::StageMemory, tile::loader::Filled};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

/// Full reader family for any precision
pub struct FullStageReaderFamily;

#[derive(CubeType)]
/// Reads any tile from the stage memory
pub struct FullStageReader<ES: Numeric, T: TilingLayout> {
    /// Stage memory from which to read
    pub stage_memory: StageMemory<ES, T>,

    #[cube(comptime)]
    /// Ident of the stage memory
    pub stage_ident: StageIdent,
}

impl StageReaderFamily for FullStageReaderFamily {
    type TileKind = Strided;
    type Reader<ES: Numeric, T: TilingLayout> = FullStageReader<ES, T>;
}

#[cube]
impl<ES: Numeric, T: TilingLayout> FullStageReader<ES, T> {
    /// Create a new FullStageToTileReader
    pub fn new(stage_memory: StageMemory<ES, T>, #[comptime] stage_ident: StageIdent) -> Self {
        FullStageReader::<ES, T> {
            stage_memory,
            stage_ident,
        }
    }
}

#[cube]
impl<ES: Numeric, T: TilingLayout> StageReader<ES> for FullStageReader<ES, T> {
    type TileKind = Strided;

    fn read_tile<S: StageMemoryConfig>(
        this: &Self,
        row: u32,
        col: u32,
        #[comptime] config: S,
    ) -> Tile<ES> {
        this.stage_memory
            .get_tile::<S>(row, col, 0u32, this.stage_ident, config)
    }
}

/// Partial reader family for any precision
pub struct PartialStageReaderFamily;

#[derive(CubeType)]
/// Reads tile from the stage memory for the specified stage ident only
pub struct PartialStageReader<ES: Numeric, T: TilingLayout> {
    pub stage_memory: StageMemory<ES, T>,
    #[cube(comptime)]
    pub stage_buffer: StageBuffer,
    #[cube(comptime)]
    stage_ident: StageIdent,
}

impl StageReaderFamily for PartialStageReaderFamily {
    type TileKind = Strided;
    type Reader<I: Numeric, T: TilingLayout> = PartialStageReader<I, T>;
}

#[cube]
impl<ES: Numeric, T: TilingLayout> PartialStageReader<ES, T> {
    /// Create a new PartialStageToTileReader
    pub fn new(
        stage_memory: StageMemory<ES, T>,
        #[comptime] stage_buffer: StageBuffer,
        #[comptime] stage_ident: StageIdent,
    ) -> PartialStageReader<ES, T> {
        PartialStageReader::<ES, T> {
            stage_memory,
            stage_buffer,
            stage_ident,
        }
    }
}

#[cube]
impl<ES: Numeric, T: TilingLayout> StageReader<ES> for PartialStageReader<ES, T> {
    type TileKind = Strided;

    fn read_tile<S: StageMemoryConfig>(
        this: &Self,
        row: u32,
        col: u32,
        #[comptime] config: S,
    ) -> Tile<ES> {
        this.stage_memory.get_tile::<S>(
            row,
            col,
            comptime!(this.stage_buffer.to_index()),
            this.stage_ident,
            config,
        )
    }
}

/// Constant value fill reader
pub struct FillStageReaderFamily;

#[derive(CubeType)]
/// Fills tile with a single value
pub struct FillStageReader<ES: Numeric> {
    value: ES,
}

impl StageReaderFamily for FillStageReaderFamily {
    type TileKind = Filled;
    type Reader<I: Numeric, T: TilingLayout> = FillStageReader<I>;
}

#[cube]
impl<ES: Numeric> FillStageReader<ES> {
    /// Create a new [`FillReader`]
    pub fn new(value: ES) -> FillStageReader<ES> {
        FillStageReader::<ES> { value }
    }
}

#[cube]
impl<ES: Numeric> StageReader<ES> for FillStageReader<ES> {
    type TileKind = Filled;

    fn read_tile<S: StageMemoryConfig>(
        this: &Self,
        _row: u32,
        _col: u32,
        #[comptime] _config: S,
    ) -> ES {
        this.value
    }
}
