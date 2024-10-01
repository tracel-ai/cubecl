use cubecl_core as cubecl;
use cubecl_core::prelude::*;

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
    const NUM_TILES_X: u32;
    const NUM_TILES_Y: u32;

    const TILE_SIZE_X: u32;
    const TILE_SIZE_Y: u32;

    fn write(writer: &mut Self, slice: &Slice<'_, E>, pos_x: u32, pos_y: u32);
    fn write_with_cast<C: Numeric>(writer: &mut Self, slice: &Slice<'_, C>, pos_x: u32, pos_y: u32);
}
