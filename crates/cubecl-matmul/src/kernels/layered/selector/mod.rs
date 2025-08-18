mod plane;
mod select_kernel;
mod unit;

pub use plane::*;
pub use select_kernel::*;
pub use unit::*;

use crate::components::{MatmulProblem, TileSize};

/// Returns  true if a [matmul problem](MatmulProblem) is very small.
///
/// A matmul is considered small based on the number of [tiles](TileSize) that fit across 2
/// dimensions, meaning that 1 dimension can still be large.
pub(crate) fn is_tiny(problem: &MatmulProblem, tile_size: &TileSize) -> bool {
    const TINY_FACTOR: usize = 2;
    const TINY_NUM_DIM: u8 = 2;

    let m = tile_size.m as usize * TINY_FACTOR >= problem.m;
    let n = tile_size.n as usize * TINY_FACTOR >= problem.n;
    let k = tile_size.k as usize * TINY_FACTOR >= problem.k;

    (m as u8 + n as u8 + k as u8) >= TINY_NUM_DIM
}
