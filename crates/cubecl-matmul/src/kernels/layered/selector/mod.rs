mod plane;
mod select_kernel;
mod unit;

pub use plane::*;
pub use select_kernel::*;
pub use unit::*;

pub(crate) fn is_tiny(
    problem: &crate::components::MatmulProblem,
    tile_size: &crate::components::TileSize,
) -> bool {
    let m = tile_size.m as usize * 2 >= problem.m;
    let n = tile_size.n as usize * 2 >= problem.n;
    let k = tile_size.k as usize * 2 >= problem.k;

    (m as u8 + n as u8 + k as u8) > 1
}
