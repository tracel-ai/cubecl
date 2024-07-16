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
    /// Bounds must be checked on lhs dimension
    pub check_m_bounds: bool,
    /// Bounds must be checked on common dimension
    pub check_k_bounds: bool,
    /// Bounds must be checked on rhs dimension
    pub check_n_bounds: bool,
    /// Tile size. Should correspond to cmma supported tile size
    pub tile_size: UInt,
    /// Vectorization of shared memory
    pub sm_vec: UInt,
    /// Lhs is transposed in global memory
    pub lhs_transposed: bool,
    /// Rhs is transposed in global memory
    pub rhs_transposed: bool,
    /// Unroll
    pub unroll: bool,
    /// Use CMMA. Otherwise falls back to slow algorithm. Deactivate for tests only
    pub use_cmma: bool,
}

impl CmmaConfig {
    pub fn new(
        m: usize,
        k: usize,
        n: usize,
        lhs_transposed: bool,
        rhs_transposed: bool,
        use_cmma: bool,
    ) -> Self {
        let block_size_m = 64;
        let block_size_k = 32;
        let block_size_n = 64;
        let tile_size = 16;
        let shared_memory_vec = 4;

        assert!(m % block_size_m == 0, "Check bounds not supported yet. ");
        assert!(k % block_size_k == 0, "Check bounds not supported yet. ");
        assert!(n % block_size_n == 0, "Check bounds not supported yet. ");

        assert!(
            !lhs_transposed,
            "Transposed input not supported yet, please make into contiguous."
        );
        assert!(
            !rhs_transposed,
            "Transposed input not supported yet, please make into contiguous."
        );

        CmmaConfig {
            block_size_m: UInt::new(block_size_m as u32),
            block_size_k: UInt::new(block_size_k as u32),
            block_size_n: UInt::new(block_size_n as u32),
            check_m_bounds: false,
            check_k_bounds: false,
            check_n_bounds: false,
            tile_size: UInt::new(tile_size),
            sm_vec: UInt::new(shared_memory_vec),
            lhs_transposed,
            rhs_transposed,
            unroll: false,
            use_cmma
        }
    }
}

pub fn cmma_cube_count<R: Runtime>(
    output_shape: &Vec<usize>,
    block_size_m: usize,
    block_size_n: usize,
) -> CubeCount<R::Server> {
    let rank = output_shape.len();
    let num_rows = *output_shape.get(rank - 2).unwrap();
    let num_cols = *output_shape.get(rank - 1).unwrap();

    let cubes_x = f32::ceil(num_rows as f32 / block_size_m as f32) as u32;
    let cubes_y = f32::ceil(num_cols as f32 / block_size_n as f32) as u32;
    let mut num_iter = 1;
    for i in 0..rank - 2 {
        num_iter *= output_shape[i];
    }

    CubeCount::Static(cubes_x, cubes_y, num_iter as u32)
}

pub fn cmma_cube_dim() -> CubeDim {
    CubeDim::new(8, 32, 1)
}
