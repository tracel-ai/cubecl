use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
/// Layout in which to store tiles within the stage
pub enum TilingOrder {
    /// Tiles are conceptually stored in row-major order, regardless of the actual data layout.
    RowMajor,
    /// Tiles are conceptually stored in column-major order, regardless of the actual data layout.
    ColMajor,
}

#[cube]
impl TilingOrder {
    /// Converts a tile index in the stage to its (x,y) position
    pub fn to_x_y(#[comptime] this: TilingOrder, nth: u32, num_x: u32, num_y: u32) -> (u32, u32) {
        match comptime!(this) {
            TilingOrder::RowMajor => (nth / num_y, nth % num_y),
            TilingOrder::ColMajor => (nth % num_x, nth / num_x),
        }
    }

    /// Converts an (x,y) position to its tile index in the stage
    pub fn to_nth_tile(
        #[comptime] this: TilingOrder,
        x: u32,
        y: u32,
        num_x: u32,
        num_y: u32,
    ) -> u32 {
        match comptime!(this) {
            TilingOrder::RowMajor => x * num_y + y,
            TilingOrder::ColMajor => y * num_x + x,
        }
    }
}
