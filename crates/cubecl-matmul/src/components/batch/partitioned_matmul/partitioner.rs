use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};
use std::fmt::Debug;
use std::hash::Hash;

use crate::components::batch::shared::swizzle;

#[cube]
/// Distributes cube instances across the tensor, assigning each to compute data in distinct regions.
pub trait Partitioner: Clone + Copy + 'static + Send + Sync + Debug + Hash + Eq {
    fn m_n_indices() -> (u32, u32);
    fn batch_index() -> u32;
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
/// Operates on data further along the m dimension as `cube_pos_x` increases,
/// and further along the n dimension as `cube_pos_y` increases.
pub struct NaturalPartitioner;

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
/// Operates on data further along the m dimension as `cube_pos_x` increases,
/// and further along the n dimension as `cube_pos_y` increases.
pub struct TransposedPartitioner;

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
/// Processes data in a swizzled pattern, prioritizing cubes along the x-axis first.
///
/// # Generics
/// - W: Width of a swizzle column
pub struct SwizzleNaturalPartitioner<const W: u32>;

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
/// Processes data in a swizzled pattern, prioritizing cubes along the y-axis first.
///
/// # Generics
/// - W: Width of a swizzle column
pub struct SwizzleTransposedPartitioner<const W: u32>;

#[cube]
impl Partitioner for NaturalPartitioner {
    fn m_n_indices() -> (u32, u32) {
        (CUBE_POS_X, CUBE_POS_Y)
    }

    fn batch_index() -> u32 {
        CUBE_POS_Z
    }

    fn create_cube_count(
        #[comptime] cubes_for_m: u32,
        #[comptime] cubes_for_n: u32,
        #[comptime] cubes_for_batches: u32,
    ) -> comptime_type!((u32, u32, u32)) {
        comptime! {(cubes_for_m, cubes_for_n, cubes_for_batches)}
    }
}

#[cube]
impl Partitioner for TransposedPartitioner {
    fn m_n_indices() -> (u32, u32) {
        (CUBE_POS_Y, CUBE_POS_X)
    }

    fn batch_index() -> u32 {
        CUBE_POS_Z
    }

    fn create_cube_count(
        #[comptime] cubes_for_m: u32,
        #[comptime] cubes_for_n: u32,
        #[comptime] cubes_for_batches: u32,
    ) -> comptime_type!((u32, u32, u32)) {
        comptime! {(cubes_for_n, cubes_for_m, cubes_for_batches)}
    }
}

#[cube]
impl<const W: u32> Partitioner for SwizzleNaturalPartitioner<W> {
    fn m_n_indices() -> (u32, u32) {
        let height = CUBE_COUNT_X;
        let nth_cube = CUBE_POS_Y * height + CUBE_POS_X;
        swizzle(nth_cube, height, W)
    }

    fn batch_index() -> u32 {
        CUBE_POS_Z
    }

    fn create_cube_count(
        #[comptime] cubes_for_m: u32,
        #[comptime] cubes_for_n: u32,
        #[comptime] cubes_for_batches: u32,
    ) -> comptime_type!((u32, u32, u32)) {
        comptime! {(cubes_for_m, cubes_for_n, cubes_for_batches)}
    }
}

#[cube]
impl<const W: u32> Partitioner for SwizzleTransposedPartitioner<W> {
    fn m_n_indices() -> (u32, u32) {
        let height = CUBE_COUNT_Y;
        let nth_cube = CUBE_POS_X * height + CUBE_POS_Y;
        swizzle(nth_cube, height, W)
    }

    fn batch_index() -> u32 {
        CUBE_POS_Z
    }

    fn create_cube_count(
        #[comptime] cubes_for_m: u32,
        #[comptime] cubes_for_n: u32,
        #[comptime] cubes_for_batches: u32,
    ) -> comptime_type!((u32, u32, u32)) {
        comptime! {(cubes_for_n, cubes_for_m, cubes_for_batches)}
    }
}
