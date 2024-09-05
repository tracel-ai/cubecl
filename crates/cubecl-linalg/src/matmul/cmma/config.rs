use cubecl_core::prelude::*;

// It is assumed that CMMA uses 32 units to compute 16x16x16 tiles
// TODO put it in config and split tile size into three different parameters
pub(crate) const CMMA_COOP_DIM: usize = 32;
pub(crate) const CMMA_TILE_SIZE: usize = 16;

pub struct CmmaBlockConfig {
    /// Corresponds to the number of tiles in the m and n dimensions for a block
    pub b_mn: usize,
    /// Corresponds to the number of tiles in the k dimension for a block
    pub b_k: usize,
    /// Corresponds to the number of accumulators per warp. Equals b_mn / b_k
    pub alpha: usize,
    /// Whether to unroll loop over k within the shared memory
    pub unroll: bool,
}

impl Default for CmmaBlockConfig {
    fn default() -> Self {
        Self::new(64, 32, false)
    }
}

impl CmmaBlockConfig {
    pub(crate) fn new(b_mn: usize, b_k: usize, unroll: bool) -> CmmaBlockConfig {
        assert!(b_mn % CMMA_TILE_SIZE == 0);
        assert!(b_k % CMMA_TILE_SIZE == 0);
        assert!(b_mn % b_k == 0);
        CmmaBlockConfig {
            b_mn,
            b_k,
            alpha: b_mn / b_k,
            unroll,
        }
    }

    pub(crate) fn comptime_info(&self, m: usize, k: usize, n: usize) -> CmmaComptimeInfo {
        let lane_dim = self.b_mn * self.b_k / (CMMA_TILE_SIZE * CMMA_TILE_SIZE);

        CmmaComptimeInfo {
            block_size_m: self.b_mn.into(),
            block_size_k: self.b_k.into(),
            block_size_n: self.b_mn.into(),
            tile_size: CMMA_TILE_SIZE.into(),
            unroll: self.unroll,
            check_m_bounds: m % self.b_mn != 0,
            check_k_bounds: k % self.b_k != 0,
            check_n_bounds: n % self.b_mn != 0,
            coop_dim: CMMA_COOP_DIM.into(),
            lane_dim: UInt::new(lane_dim as u32),
            num_accumulators: UInt::new(self.alpha as u32),
        }
    }

    pub(crate) fn cube_count<R: Runtime>(
        &self,
        output_shape: &[usize],
    ) -> CubeCount<<R as Runtime>::Server> {
        let rank = output_shape.len();
        let num_rows = *output_shape.get(rank - 2).unwrap();
        let num_cols = *output_shape.get(rank - 1).unwrap();

        let cubes_x = f32::ceil(num_rows as f32 / self.b_mn as f32) as u32;
        let cubes_y = f32::ceil(num_cols as f32 / self.b_mn as f32) as u32;

        let mut num_iter = 1;
        for shape in output_shape.iter().take(rank - 2) {
            num_iter *= shape;
        }

        CubeCount::Static(cubes_x, cubes_y, num_iter as u32)
    }

    pub(crate) fn cube_dim(&self) -> CubeDim {
        // A bit arbitrary as long as number of elements stays the same
        // TODO allow trying other combinations that have same product
        CubeDim {
            x: CMMA_COOP_DIM as u32,
            y: ((self.b_mn * self.b_k) / (CMMA_TILE_SIZE * CMMA_TILE_SIZE)) as u32,
            z: 1,
        }
    }

    pub(crate) fn available_vectorizations(&self) -> Vec<u8> {
        let vectorizations = vec![8, 4, 2];
        for v in vectorizations.iter() {
            assert!(*v as usize * CMMA_COOP_DIM % (CMMA_TILE_SIZE * CMMA_TILE_SIZE) == 0);
        }
        vectorizations
    }
}

impl Init for CmmaComptimeInfo {
    fn init(self, _context: &mut CubeContext) -> Self {
        self
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
/// Tiling 2D parameters
pub struct CmmaComptimeInfo {
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
    /// The number of units that can collaborate
    pub coop_dim: UInt,
    /// The number of collaboration groups
    pub lane_dim: UInt,
    /// Number of cmma per subcube performed in one pass
    pub num_accumulators: UInt,
}
