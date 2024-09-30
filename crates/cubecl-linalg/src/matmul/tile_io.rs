use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::MatmulInstruction;

#[cube]
/// Defines the number of tiles and their size in each plane
pub trait TileReader<E: CubeType>: CubeType + 'static + Send + Sync {
    /// Number of tiles, vertically
    const NUM_TILES_X: u32;
    /// Number of tiles, horizontally
    const NUM_TILES_Y: u32;

    /// Size of each tile, vertically
    const TILE_SIZE_X: u32;
    /// Size of each tile, horizontally
    const TILE_SIZE_Y: u32;

    /// Return the tile at position (x,y) as a contiguous slice
    fn read(reader: &Self, pos_x: u32, pos_y: u32) -> &Slice<'_, E>;
}

#[cube]
pub trait MatmulInstructionWriter: CubeType {}

#[cube]
/// Defines the number of tiles and their size in each plane
pub trait TileWriter<E: CubeType>: CubeType + 'static + Send + Sync {
    // TODO: wrong that it uselessly needs to know I and O
    fn from_instruction_to_output<'a, Instr: MatmulInstruction<I, O>, I: Numeric, O: Numeric>(
        writer: &'a mut Self,
        instr_out: &Instr::Out,
        pos_x: u32,
        pos_y: u32,
    ) -> &'a mut SliceMut<'a, E>;
}
