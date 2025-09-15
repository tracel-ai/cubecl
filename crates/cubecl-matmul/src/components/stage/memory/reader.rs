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
pub struct FullReaderFamily;

#[derive(CubeType)]
/// Reads any tile from the stage memory
pub struct FullStageToTileReader<ES: Numeric, T: TilingLayout> {
    /// Stage memory from which to read
    pub stage_memory: StageMemory<ES, T>,

    #[cube(comptime)]
    /// Ident of the stage memory
    pub stage_ident: StageIdent,
}

impl StageReaderFamily for FullReaderFamily {
    type TileKind = Strided;
    type Reader<ES: Numeric, T: TilingLayout> = FullStageToTileReader<ES, T>;
}

#[cube]
impl<ES: Numeric, T: TilingLayout> FullStageToTileReader<ES, T> {
    /// Create a new FullStageToTileReader
    pub fn new(stage_memory: StageMemory<ES, T>, #[comptime] stage_ident: StageIdent) -> Self {
        FullStageToTileReader::<ES, T> {
            stage_memory,
            stage_ident,
        }
    }
}

#[cube]
impl<ES: Numeric, T: TilingLayout> StageReader<ES> for FullStageToTileReader<ES, T> {
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
pub struct PartialReaderFamily;

#[derive(CubeType)]
/// Reads tile from the stage memory for the specified stage ident only
pub struct PartialStageToTileReader<ES: Numeric, T: TilingLayout> {
    pub stage_memory: StageMemory<ES, T>,
    #[cube(comptime)]
    pub stage_buffer: StageBuffer,
    #[cube(comptime)]
    stage_ident: StageIdent,
}

impl StageReaderFamily for PartialReaderFamily {
    type TileKind = Strided;
    type Reader<I: Numeric, T: TilingLayout> = PartialStageToTileReader<I, T>;
}

#[cube]
impl<ES: Numeric, T: TilingLayout> PartialStageToTileReader<ES, T> {
    /// Create a new PartialStageToTileReader
    pub fn new(
        stage_memory: StageMemory<ES, T>,
        #[comptime] stage_buffer: StageBuffer,
        #[comptime] stage_ident: StageIdent,
    ) -> PartialStageToTileReader<ES, T> {
        PartialStageToTileReader::<ES, T> {
            stage_memory,
            stage_buffer,
            stage_ident,
        }
    }
}

#[cube]
impl<ES: Numeric, T: TilingLayout> StageReader<ES> for PartialStageToTileReader<ES, T> {
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
pub struct FillReaderFamily;

#[derive(CubeType)]
/// Fills tile with a single value
pub struct FillReader<ES: Numeric> {
    value: ES,
}

impl StageReaderFamily for FillReaderFamily {
    type TileKind = Filled;
    type Reader<I: Numeric, T: TilingLayout> = FillReader<I>;
}

#[cube]
impl<ES: Numeric> FillReader<ES> {
    /// Create a new [`FillReader`]
    pub fn new(value: ES) -> FillReader<ES> {
        FillReader::<ES> { value }
    }
}

#[cube]
impl<ES: Numeric> StageReader<ES> for FillReader<ES> {
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
