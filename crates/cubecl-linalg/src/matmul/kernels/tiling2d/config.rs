use cubecl_core::{
    self as cubecl, CubeDim,
    prelude::{IntoMut, Scope},
};
use cubecl_core::{CubeCount, CubeType};

use super::base::TILE_SIZE;

#[derive(Debug, Clone)]
/// Tiling 2D parameters
pub struct Tiling2dConfig {
    /// Block size along dimension of lhs
    pub block_size_m: usize,
    /// Block size along common dimension
    pub block_size_k: usize,
    /// Block size along dimension of rhs
    pub block_size_n: usize,
    /// Tile size and shared memory vectorization
    pub tile_size: usize,
    /// Loop unrolling
    pub unroll: bool,
}

impl Default for Tiling2dConfig {
    fn default() -> Self {
        Self {
            block_size_m: 64,
            block_size_k: 32,
            block_size_n: 64,
            tile_size: TILE_SIZE,
            unroll: false,
        }
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug, CubeType)]
/// Tiling 2D parameters
pub struct CubeTiling2dConfig {
    /// Block size along dimension of lhs
    pub block_size_m: u32,
    /// Block size along common dimension
    pub block_size_k: u32,
    /// Block size along dimension of rhs
    pub block_size_n: u32,
    /// Loop unrolling for inner compute loop. Probably slower
    pub unroll_compute: bool,
    /// Loop unrolling for all loops related to vectorization/tile size. Probably faster
    pub unroll_tile: bool,
    /// Bounds must be checked on lhs dimension
    pub check_m_bounds: bool,
    /// Bounds must be checked on common dimension
    pub check_k_bounds: bool,
    /// Bounds must be checked on rhs dimension
    pub check_n_bounds: bool,
    /// Tile size. Should correspond to vectorization of inputs/outputs/shared memory
    pub tile_size: u32,
    /// Lhs is transposed in global memory
    pub lhs_transposed: bool,
    /// Rhs is transposed in global memory
    pub rhs_transposed: bool,
}

impl IntoMut for CubeTiling2dConfig {
    fn into_mut(self, _scope: &mut Scope) -> Self {
        self
    }
}

impl CubeTiling2dConfig {
    pub fn new(
        config: &Tiling2dConfig,
        m: usize,
        k: usize,
        n: usize,
        lhs_transposed: bool,
        rhs_transposed: bool,
    ) -> Self {
        assert!(
            config.block_size_k <= config.block_size_m
                && config.block_size_k <= config.block_size_n,
            "Larger block size in k than m or n results in unfilled shared memory."
        );
        assert!(
            config.block_size_m % config.tile_size == 0
                && config.block_size_k % config.tile_size == 0
                && config.block_size_n % config.tile_size == 0,
            "Tiling 2d algorithm assumes tile size divides block size perfectly. "
        );

        CubeTiling2dConfig {
            block_size_m: config.block_size_m as u32,
            block_size_k: config.block_size_k as u32,
            block_size_n: config.block_size_n as u32,
            unroll_compute: config.unroll,
            unroll_tile: true,
            check_m_bounds: m % config.block_size_m != 0,
            check_k_bounds: k % config.block_size_k != 0,
            check_n_bounds: n % config.block_size_n != 0,
            tile_size: config.tile_size as u32,
            lhs_transposed,
            rhs_transposed,
        }
    }
}

pub fn tiling2d_cube_count(output_shape: &[usize], config: &Tiling2dConfig) -> CubeCount {
    let rank = output_shape.len();
    let num_rows = *output_shape.get(rank - 2).unwrap();
    let num_cols = *output_shape.get(rank - 1).unwrap();

    let cubes_x = f32::ceil(num_rows as f32 / config.block_size_m as f32) as u32;
    let cubes_y = f32::ceil(num_cols as f32 / config.block_size_n as f32) as u32;
    let mut num_iter = 1;
    for shape in output_shape.iter().take(rank - 2) {
        num_iter *= shape;
    }

    CubeCount::Static(cubes_x, cubes_y, num_iter as u32)
}

pub fn tiling2d_cube_dim(config: &Tiling2dConfig) -> CubeDim {
    CubeDim::new(
        (config.block_size_m / config.tile_size) as u32,
        (config.block_size_n / config.tile_size) as u32,
        1,
    )
}
