use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};
use std::fmt::Debug;
use std::hash::Hash;

use crate::components::batch::shared::swizzle;

#[cube]
/// Distributes cube instances across the tensor, assigning each to compute data in distinct regions.
pub trait CubeDispatch: Clone + Copy + 'static + Send + Sync + Debug + Hash + Eq {
    fn x_y_indices() -> (u32, u32);
    fn batch_index() -> u32;
    fn max_x(#[comptime] cube_count: (u32, u32, u32)) -> comptime_type!(u32);
    fn max_y(#[comptime] cube_count: (u32, u32, u32)) -> comptime_type!(u32);
    fn max_batches(#[comptime] cube_count: (u32, u32, u32)) -> comptime_type!(u32);
}

pub trait CubeCountDispatch {
    fn cube_count(cubes_for_m: u32, cubes_for_n: u32, cubes_for_batches: u32) -> CubeCount;
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
pub struct SwizzleNaturalDispatch<const W: u32>;

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
/// Processes data in a swizzled pattern, prioritizing cubes along the y-axis first.
///
/// # Generics
/// - W: Width of a swizzle column
pub struct SwizzleTransposedDispatch<const W: u32>;

#[cube]
impl CubeDispatch for NaturalDispatch {
    fn x_y_indices() -> (u32, u32) {
        (CUBE_POS_X, CUBE_POS_Y)
    }

    fn batch_index() -> u32 {
        CUBE_POS_Z
    }

    fn max_x(#[comptime] cube_count: (u32, u32, u32)) -> comptime_type!(u32) {
        cube_count.0
    }

    fn max_y(#[comptime] cube_count: (u32, u32, u32)) -> comptime_type!(u32) {
        cube_count.1
    }

    fn max_batches(#[comptime] cube_count: (u32, u32, u32)) -> comptime_type!(u32) {
        cube_count.2
    }
}

impl CubeCountDispatch for NaturalDispatch {
    fn cube_count(cubes_for_m: u32, cubes_for_n: u32, cubes_for_batches: u32) -> CubeCount {
        CubeCount::Static(cubes_for_m, cubes_for_n, cubes_for_batches)
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

    fn max_x(#[comptime] cube_count: (u32, u32, u32)) -> comptime_type!(u32) {
        cube_count.1
    }

    fn max_y(#[comptime] cube_count: (u32, u32, u32)) -> comptime_type!(u32) {
        cube_count.0
    }

    fn max_batches(#[comptime] cube_count: (u32, u32, u32)) -> comptime_type!(u32) {
        cube_count.2
    }
}

impl CubeCountDispatch for TransposedDispatch {
    fn cube_count(cubes_for_m: u32, cubes_for_n: u32, cubes_for_batches: u32) -> CubeCount {
        CubeCount::Static(cubes_for_n, cubes_for_m, cubes_for_batches)
    }
}

#[cube]
impl<const W: u32> CubeDispatch for SwizzleNaturalDispatch<W> {
    fn x_y_indices() -> (u32, u32) {
        let height = CUBE_COUNT_X;
        let nth_cube = CUBE_POS_Y * height + CUBE_POS_X;
        swizzle(nth_cube, height, W)
    }

    fn batch_index() -> u32 {
        CUBE_POS_Z
    }

    fn max_x(#[comptime] cube_count: (u32, u32, u32)) -> comptime_type!(u32) {
        cube_count.0
    }

    fn max_y(#[comptime] cube_count: (u32, u32, u32)) -> comptime_type!(u32) {
        cube_count.1
    }

    fn max_batches(#[comptime] cube_count: (u32, u32, u32)) -> comptime_type!(u32) {
        cube_count.2
    }
}

impl<const W: u32> CubeCountDispatch for SwizzleNaturalDispatch<W> {
    fn cube_count(cubes_for_m: u32, cubes_for_n: u32, cubes_for_batches: u32) -> CubeCount {
        CubeCount::Static(cubes_for_m, cubes_for_n, cubes_for_batches)
    }
}

#[cube]
impl<const W: u32> CubeDispatch for SwizzleTransposedDispatch<W> {
    fn x_y_indices() -> (u32, u32) {
        let height = CUBE_COUNT_Y;
        let nth_cube = CUBE_POS_X * height + CUBE_POS_Y;
        swizzle(nth_cube, height, W)
    }

    fn batch_index() -> u32 {
        CUBE_POS_Z
    }

    fn max_x(#[comptime] cube_count: (u32, u32, u32)) -> comptime_type!(u32) {
        cube_count.1
    }

    fn max_y(#[comptime] cube_count: (u32, u32, u32)) -> comptime_type!(u32) {
        cube_count.0
    }

    fn max_batches(#[comptime] cube_count: (u32, u32, u32)) -> comptime_type!(u32) {
        cube_count.2
    }
}

impl<const W: u32> CubeCountDispatch for SwizzleTransposedDispatch<W> {
    fn cube_count(cubes_for_m: u32, cubes_for_n: u32, cubes_for_batches: u32) -> CubeCount {
        CubeCount::Static(cubes_for_n, cubes_for_m, cubes_for_batches)
    }
}
