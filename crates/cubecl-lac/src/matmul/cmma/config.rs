use cubecl_core::prelude::*;

impl Init for CmmaConfig {
    fn init(self, _context: &mut CubeContext) -> Self {
        self
    }
}

#[derive(Debug, Clone, Copy)]
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
    pub _check_m_bounds: bool,
    /// Bounds must be checked on common dimension
    pub _check_k_bounds: bool,
    /// Bounds must be checked on rhs dimension
    pub _check_n_bounds: bool,
    /// Unroll
    pub unroll: bool,
}

impl Default for CmmaConfig {
    fn default() -> Self {
        Self {
            block_size_m: UInt::new(64),
            block_size_k: UInt::new(32),
            block_size_n: UInt::new(64),
            tile_size: UInt::new(16),
            _check_m_bounds: false,
            _check_k_bounds: false,
            _check_n_bounds: false,
            unroll: false,
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
