use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, CubeType};

use super::{block_loop::block_loop, config::CubeTiling2dConfig};

/// Most common tile size, the one used in most tests.
pub(crate) const TILE_SIZE: usize = 4;

#[cube(launch_unchecked)]
#[allow(unused_mut)]
pub fn tiling2d_cube_kernel<N: Numeric>(
    lhs: &Tensor<Line<N>>,
    rhs: &Tensor<Line<N>>,
    out: &mut Tensor<Line<N>>,
    #[comptime] config: CubeTiling2dConfig,
) {
    let dims = get_dims::<N>(lhs, rhs);
    let coordinates = calculate_coordinates(CUBE_POS_X, CUBE_POS_Y, UNIT_POS, config);
    let offsets = calculate_batch_offsets::<N>(lhs, rhs, out, CUBE_POS_Z);
    let shared_memories = make_shared_memories::<N>(config);

    block_loop::<N>(
        lhs,
        rhs,
        out,
        coordinates,
        offsets,
        shared_memories,
        config,
        dims,
    );
}

#[derive(CubeType, Copy, Clone)]
/// Information available at runtime only
/// Strides assume contiguous
pub(crate) struct Dimensions {
    pub m: u32,
    pub k: u32,
    pub n: u32,
}

#[derive(CubeType, Copy, Clone)]
pub(crate) struct SharedMemories<N: Numeric> {
    pub lhs: SharedMemory<Line<N>>,
    pub rhs: SharedMemory<Line<N>>,
}

#[derive(CubeType, Copy, Clone)]
/// Number of elements in previous batches
/// Not divided by vectorization facto
pub(crate) struct BatchOffsets {
    pub lhs: u32,
    pub rhs: u32,
    pub out: u32,
}

#[derive(CubeType, Copy, Clone)]
pub(crate) struct Coordinates {
    pub unit_row: u32,
    pub unit_col: u32,
    pub skip_row: u32,
    pub skip_col: u32,
}

#[cube]
fn get_dims<N: Numeric>(lhs: &Tensor<Line<N>>, rhs: &Tensor<Line<N>>) -> Dimensions {
    let rank = lhs.rank();
    let first_dim = rank - 2;
    let second_dim = rank - 1;
    let m = lhs.shape(first_dim);
    let k = lhs.shape(second_dim);
    let n = rhs.shape(second_dim);

    Dimensions { m, k, n }
}

#[cube]
fn calculate_coordinates(
    cube_pos_x: u32,
    cube_pos_y: u32,
    unit_pos: u32,
    #[comptime] config: CubeTiling2dConfig,
) -> Coordinates {
    let block_size_m = config.block_size_m;
    let block_size_n = config.block_size_n;
    let tile_size = config.tile_size;

    let n_units_per_row = ((block_size_n - 1) / tile_size) + 1;

    // Cube offset
    let skip_row = cube_pos_x * block_size_m;
    let skip_col = cube_pos_y * block_size_n;

    // Position of the first element of the unit, relative to the cube
    let unit_row = (unit_pos / n_units_per_row) * tile_size;
    let unit_col = (unit_pos % n_units_per_row) * tile_size;

    Coordinates {
        unit_row,
        unit_col,
        skip_row,
        skip_col,
    }
}

#[cube]
#[allow(unused_mut)]
fn calculate_batch_offsets<N: Numeric>(
    lhs: &Tensor<Line<N>>,
    rhs: &Tensor<Line<N>>,
    out: &Tensor<Line<N>>,
    batch_number: u32,
) -> BatchOffsets {
    let rank = out.rank();

    // Batch offset for output
    let mut offset_out = batch_number * out.stride(rank - 2) * out.shape(rank - 2);
    let mut offset_lhs = 0;
    let mut offset_rhs = 0;

    // Batch offset for lhs, rhs
    for b in 0..rank - 2 {
        let tmp = offset_out / out.stride(b);
        offset_lhs += tmp % lhs.shape(b) * lhs.stride(b);
        offset_rhs += tmp % rhs.shape(b) * rhs.stride(b);
    }

    BatchOffsets {
        lhs: offset_lhs,
        rhs: offset_rhs,
        out: offset_out,
    }
}

#[cube]
fn make_shared_memories<N: Numeric>(#[comptime] config: CubeTiling2dConfig) -> SharedMemories<N> {
    let tile_size = config.tile_size;
    let block_size_m = config.block_size_m;
    let block_size_k = config.block_size_k;
    let block_size_n = config.block_size_n;

    let lhs = SharedMemory::<N>::new_lined(block_size_k * block_size_m / tile_size, tile_size);
    let rhs = SharedMemory::<N>::new_lined(block_size_k * block_size_n / tile_size, tile_size);

    SharedMemories::<N> { lhs, rhs }
}
