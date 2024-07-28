use cubecl_core::prelude::*;

impl Init for CmmaConfig {
    fn init(self, _context: &mut CubeContext) -> Self {
        self
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
/// Tiling 2D parameters
pub struct CmmaConfig {
    /// Block size along dimension of lhs
    pub block_size_m: UInt,
    /// Block size along common dimension
    pub block_size_k: UInt,
    /// Block size along dimension of rhs
    pub block_size_n: UInt,
    /// Tile size (dimension of one side). Should correspond to cmma supported tile size
    pub tile_size: UInt,
    /// Bounds must be checked on lhs dimension
    pub check_m_bounds: bool,
    /// Bounds must be checked on common dimension
    pub check_k_bounds: bool,
    /// Bounds must be checked on rhs dimension
    pub check_n_bounds: bool,
    /// Unroll
    pub unroll: bool,
}

pub struct CmmaLaunchConfig {
    /// Block size along dimension of lhs
    pub block_size_m: usize,
    /// Block size along common dimension
    pub block_size_k: usize,
    /// Block size along dimension of rhs
    pub block_size_n: usize,
    /// Tile size (dimension of one side). Should correspond to cmma supported tile size
    pub tile_size: usize,
    /// Unroll
    pub unroll: bool,
}

impl Default for CmmaLaunchConfig {
    fn default() -> Self {
        Self {
            block_size_m: 64,
            block_size_k: 32,
            block_size_n: 64,
            tile_size: 16,
            unroll: false,
        }
    }
}

impl CmmaConfig {
    pub(crate) fn new(m: usize, k: usize, n: usize, launch_config: CmmaLaunchConfig) -> Self {
        CmmaConfig {
            block_size_m: launch_config.block_size_m.into(),
            block_size_k: launch_config.block_size_k.into(),
            block_size_n: launch_config.block_size_n.into(),
            tile_size: launch_config.tile_size.into(),
            unroll: launch_config.unroll,
            check_m_bounds: m % launch_config.block_size_m != 0,
            check_k_bounds: k % launch_config.block_size_k != 0,
            check_n_bounds: n % launch_config.block_size_n != 0,
        }
    }
}

pub fn cmma_cube_count<R: Runtime>(
    output_shape: &[usize],
    block_size_m: usize,
    block_size_n: usize,
) -> CubeCount<R::Server> {
    let rank = output_shape.len();
    let num_rows = *output_shape.get(rank - 2).unwrap();
    let num_cols = *output_shape.get(rank - 1).unwrap();

    let cubes_x = f32::ceil(num_rows as f32 / block_size_m as f32) as u32;
    let cubes_y = f32::ceil(num_cols as f32 / block_size_n as f32) as u32;
    let mut num_iter = 1;
    for shape in output_shape.iter().take(rank - 2) {
        num_iter *= shape;
    }

    CubeCount::Static(cubes_x, cubes_y, num_iter as u32)
}

pub fn cmma_cube_dim() -> CubeDim {
    CubeDim::new(32, 8, 1)
}
