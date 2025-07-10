use crate::components::InputIdent;
use crate::components::global::load::StageIdent;
use crate::components::stage::ReaderFamily;
use crate::components::stage::StageConfig;
use crate::components::stage::StageMemory;
use crate::components::stage::StageToTileReader;
use crate::components::stage::TilingLayout;
use crate::components::tile::Tile;
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
    pub input_ident: InputIdent,
}

impl ReaderFamily for FullReaderFamily {
    type Reader<ES: Numeric, T: TilingLayout> = FullStageToTileReader<ES, T>;
}

#[cube]
impl<ES: Numeric, T: TilingLayout> FullStageToTileReader<ES, T> {
    /// Create a new FullStageToTileReader
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

/// Partial reader family for any precision
pub struct PartialReaderFamily;

#[derive(CubeType)]
/// Reads tile from the stage memory for the specified stage ident only
pub struct PartialStageToTileReader<ES: Numeric, T: TilingLayout> {
    pub stage_memory: StageMemory<ES, T>,
    #[cube(comptime)]
    pub stage_ident: StageIdent,
    #[cube(comptime)]
    input_ident: InputIdent,
}

impl ReaderFamily for PartialReaderFamily {
    type Reader<I: Numeric, T: TilingLayout> = PartialStageToTileReader<I, T>;
}

#[cube]
impl<ES: Numeric, T: TilingLayout> PartialStageToTileReader<ES, T> {
    /// Create a new PartialStageToTileReader
    pub fn new(
        stage_memory: StageMemory<ES, T>,
        #[comptime] stage_ident: StageIdent,
        #[comptime] input_ident: InputIdent,
    ) -> PartialStageToTileReader<ES, T> {
        PartialStageToTileReader::<ES, T> {
            stage_memory,
            stage_ident,
            input_ident,
        }
    }
}

#[cube]
impl<ES: Numeric, T: TilingLayout> StageToTileReader<ES> for PartialStageToTileReader<ES, T> {
    fn read_tile<S: StageConfig>(
        this: &Self,
        row: u32,
        col: u32,
        #[comptime] config: S,
    ) -> Tile<ES> {
        this.stage_memory.get_tile::<S>(
            row,
            col,
            comptime!(this.stage_ident.to_index()),
            comptime!(this.input_ident.as_ident()),
            config,
        )
    }
}
