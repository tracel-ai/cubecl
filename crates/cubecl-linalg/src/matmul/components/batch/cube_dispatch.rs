use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};
use std::fmt::Debug;
use std::hash::Hash;

use crate::matmul::components::batch::shared::swizzle;

#[cube]
pub trait CubeDispatch: Clone + Copy + 'static + Send + Sync + Debug + Hash + Eq {
    fn x_y_indices() -> (u32, u32);
    fn batch_index() -> u32;
    fn max_x(#[comptime] cube_count: (u32, u32, u32)) -> u32;
    fn max_y(#[comptime] cube_count: (u32, u32, u32)) -> u32;
    fn max_batches(#[comptime] cube_count: (u32, u32, u32)) -> u32;
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
/// Operates on data further along the m dimension as `cube_pos_x` increases,
/// and further along the n dimension as `cube_pos_y` increases.
pub struct NaturalDispatch;

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
/// Operates on data further along the m dimension as `cube_pos_x` increases,
/// and further along the n dimension as `cube_pos_y` increases.
pub struct TransposedDispatch;

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
/// Processes data in a swizzled pattern, prioritizing cubes along the x-axis first.
///
/// # Generics
/// - W: Width of a swizzle column
/// - H: Height of swizzle columns (number of rows)
pub struct SwizzleXFirstDispatch<const W: u32, const H: u32>;

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
/// Processes data in a swizzled pattern, prioritizing cubes along the y-axis first.
///
/// # Generics
/// - W: Width of a swizzle column
/// - H: Height of swizzle columns (number of rows)
pub struct SwizzleYFirstDispatch<const W: u32, const H: u32>;

#[cube]
impl CubeDispatch for NaturalDispatch {
    fn x_y_indices() -> (u32, u32) {
        (CUBE_POS_X, CUBE_POS_Y)
    }

    fn batch_index() -> u32 {
        CUBE_POS_Z
    }

    fn max_x(#[comptime] cube_count: (u32, u32, u32)) -> u32 {
        cube_count.0
    }

    fn max_y(#[comptime] cube_count: (u32, u32, u32)) -> u32 {
        cube_count.1
    }

    fn max_batches(#[comptime] cube_count: (u32, u32, u32)) -> u32 {
        cube_count.2
    }
}

#[cube]
impl CubeDispatch for TransposedDispatch {
    fn x_y_indices() -> (u32, u32) {
        (CUBE_POS_Y, CUBE_POS_X)
    }

    fn batch_index() -> u32 {
        CUBE_POS_Z
    }

    fn max_x(#[comptime] cube_count: (u32, u32, u32)) -> u32 {
        cube_count.1
    }

    fn max_y(#[comptime] cube_count: (u32, u32, u32)) -> u32 {
        cube_count.0
    }

    fn max_batches(#[comptime] cube_count: (u32, u32, u32)) -> u32 {
        cube_count.2
    }
}

#[cube]
impl<const W: u32, const H: u32> CubeDispatch for SwizzleXFirstDispatch<W, H> {
    fn x_y_indices() -> (u32, u32) {
        let nth_cube = CUBE_POS_Y * CUBE_DIM_X + CUBE_POS_X;
        swizzle(nth_cube, H, W)
    }

    fn batch_index() -> u32 {
        CUBE_POS_Z
    }

    fn max_x(#[comptime] cube_count: (u32, u32, u32)) -> u32 {
        cube_count.0
    }

    fn max_y(#[comptime] cube_count: (u32, u32, u32)) -> u32 {
        cube_count.1
    }

    fn max_batches(#[comptime] cube_count: (u32, u32, u32)) -> u32 {
        cube_count.2
    }
}

#[cube]
impl<const W: u32, const H: u32> CubeDispatch for SwizzleYFirstDispatch<W, H> {
    fn x_y_indices() -> (u32, u32) {
        let nth_cube = CUBE_POS_X * CUBE_DIM_Y + CUBE_POS_Y;
        swizzle(nth_cube, H, W)
    }

    fn batch_index() -> u32 {
        CUBE_POS_Z
    }

    fn max_x(#[comptime] cube_count: (u32, u32, u32)) -> u32 {
        cube_count.1
    }

    fn max_y(#[comptime] cube_count: (u32, u32, u32)) -> u32 {
        cube_count.0
    }

    fn max_batches(#[comptime] cube_count: (u32, u32, u32)) -> u32 {
        cube_count.2
    }
}
