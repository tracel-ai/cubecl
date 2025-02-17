use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// How the tiles are stored in shared memory
pub enum TilingLayout {
    /// Each tile is stored contiguously in memory.
    /// Tiles are placed sequentially in memory according to the specified `TilingOrder`.
    Contiguous(TilingOrder),

    /// Tiles follow the memory layout of the underlying global memory,
    /// meaning elements within a tile may be interleaved with elements from other tiles.
    Strided,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
/// Layout in which to store tiles within the stage
pub enum TilingOrder {
    /// Tiles are conceptually stored in row-major order, regardless of the actual data layout.
    RowMajor,
    /// Tiles are conceptually stored in column-major order, regardless of the actual data layout.
    ColMajor,
}

#[cube]
impl TilingLayout {
    /// Converts a tile index in the stage to its (x,y) position
    pub fn to_x_y(#[comptime] this: TilingLayout, nth: u32, num_x: u32, num_y: u32) -> (u32, u32) {
        match comptime!(this) {
            TilingLayout::Contiguous(tiling_order) => match comptime!(tiling_order) {
                TilingOrder::RowMajor => (nth / num_y, nth % num_y),
                TilingOrder::ColMajor => (nth % num_x, nth / num_x),
            },
            TilingLayout::Strided => todo!(),
        }
    }

    /// Converts an (x,y) position to its tile index in the stage
    pub fn to_nth_tile(
        #[comptime] this: TilingLayout,
        x: u32,
        y: u32,
        num_x: u32,
        num_y: u32,
    ) -> u32 {
        match comptime!(this) {
            TilingLayout::Contiguous(tiling_order) => match comptime!(tiling_order) {
                TilingOrder::RowMajor => x * num_y + y,
                TilingOrder::ColMajor => y * num_x + x,
            },
            TilingLayout::Strided => todo!(),
        }
    }
}
