use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

#[cube]
/// Layout in which to store tiles within the stage
pub trait TilingOrder: Clone + Copy + 'static + Send + Sync {
    /// Converts a tile index in the stage to its (x,y) position
    fn to_x_y(nth: u32, num_x: u32, num_y: u32) -> (u32, u32);
    /// Converts an (x,y) position to its tile index in the stage
    fn to_nth_tile(x: u32, y: u32, num_x: u32, num_y: u32) -> u32;
}

#[derive(CubeType, Copy, Clone, PartialEq, Eq, Hash, Debug)]
/// Config determining which existing TilingOrder to use
pub enum TilingOrderConfig {
    XMajor,
    YMajor,
}

#[derive(Clone, Copy)]
/// Tiles are conceptually stored in row-major order, regardless of the actual data layout.
pub struct XMajorTiling {}
#[derive(Clone, Copy)]
/// Tiles are conceptually stored in column-major order, regardless of the actual data layout.
pub struct YMajorTiling {}

#[cube]
impl TilingOrder for XMajorTiling {
    fn to_x_y(nth: u32, _num_x: u32, num_y: u32) -> (u32, u32) {
        (nth / num_y, nth % num_y)
    }

    fn to_nth_tile(x: u32, y: u32, _num_x: u32, num_y: u32) -> u32 {
        x * num_y + y
    }
}

#[cube]
impl TilingOrder for YMajorTiling {
    fn to_x_y(nth: u32, num_x: u32, _num_y: u32) -> (u32, u32) {
        (nth % num_x, nth / num_x)
    }

    fn to_nth_tile(x: u32, y: u32, num_x: u32, _num_y: u32) -> u32 {
        y * num_x + x
    }
}
