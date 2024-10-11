use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

#[cube]
pub trait TilingOrder: Clone + Copy + 'static + Send + Sync {
    fn to_x_y(nth: u32, num_x: u32, num_y: u32) -> (u32, u32);
    fn to_nth_tile(x: u32, y: u32, num_x: u32, num_y: u32) -> u32;
}

#[derive(Clone, Copy)]
pub struct XMajorTiling {}
#[derive(Clone, Copy)]
pub struct YMajorTiling {}

#[cube]
impl TilingOrder for XMajorTiling {
    fn to_x_y(nth_tile: u32, _num_x: u32, num_y: u32) -> (u32, u32) {
        (nth_tile / num_y, nth_tile % num_y)
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
