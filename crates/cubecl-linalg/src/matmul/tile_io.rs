use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
/// Defines the number of tiles and their size in each plane
pub trait TileReader<E: CubeType>: CubeType {
    /// Number of tiles, vertically
    const NUM_TILES_X: u32;
    /// Number of tiles, horizontally
    const NUM_TILES_Y: u32;

    /// Size of each tile, vertically
    const TILE_SIZE_X: u32;
    /// Size of each tile, horizontally
    const TILE_SIZE_Y: u32;

    /// Return the tile at position (x,y) as a contiguous slice
    fn read(reader: &Self, pos_x: u32, pos_y: u32) -> Slice<'_, E>;
}

#[cube]
/// Defines the number of tiles and their size in each plane
pub trait TileWriter<E: CubeType>: CubeType {
    fn get_tile_as_slice_mut(writer: &Self, pos_x: u32, pos_y: u32) -> SliceMut<'_, E>;
    fn reorganize_slice(writer: &Self, slice: &Slice<'_, E>, pos_x: u32, pos_y: u32);
}
