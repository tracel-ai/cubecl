use cubecl_core::prelude::*;

// It is assumed that CMMA uses 32 units to compute 16x16x16 tiles
// TODO put it in config and split tile size into three different parameters
pub(crate) const CMMA_COOP_DIM: usize = 32;
pub(crate) const CMMA_TILE_SIZE: usize = 16;

#[derive(PartialEq, Eq)]
/// Defines how data travels from accumulators to global output
pub enum WriteOutStrategy {
    /// Accumulators for one warp are put concurrently in a shared memory large enough to contain them all
    LargeSmem,
    /// Accumulators for one warp are put sequentially in a shared memory with only one reusable spot
    ReuseSmem,
}

pub struct CmmaConfig {
    /// Corresponds to the number of tiles in the m and n dimensions for a block
    pub b_mn: usize,
    /// Corresponds to the number of tiles in the k dimension for a block
    pub b_k: usize,
    /// Whether to unroll loop over k within the shared memory
    pub unroll: bool,
    /// Whether to write all accumulators in different spots of a large shared memory or reuse the space
    pub write_out_strategy: WriteOutStrategy,
    /// Corresponds to the number of accumulators per warp. Equals b_mn / b_k
    pub alpha: usize,
}

impl Default for CmmaConfig {
    fn default() -> Self {
        Self::new(128, 16, false, WriteOutStrategy::ReuseSmem)
    }
}

impl CmmaConfig {
    pub(crate) fn new(
        b_mn: usize,
        b_k: usize,
        unroll: bool,
        write_out_strategy: WriteOutStrategy,
    ) -> CmmaConfig {
        assert!(b_mn % CMMA_TILE_SIZE == 0);
        assert!(b_k % CMMA_TILE_SIZE == 0);
        assert!(b_mn % b_k == 0);
        CmmaConfig {
            b_mn,
            b_k,
            alpha: b_mn / b_k,
            unroll,
            write_out_strategy,
        }
    }

    pub(crate) fn comptime_info(&self, m: usize, k: usize, n: usize) -> ComptimeCmmaInfo {
        let num_coops = self.b_mn * self.b_k / (CMMA_TILE_SIZE * CMMA_TILE_SIZE);

        ComptimeCmmaInfo {
            block_size_m: self.b_mn.into(),
            block_size_k: self.b_k.into(),
            block_size_n: self.b_mn.into(),
            tile_size: CMMA_TILE_SIZE.into(),
            unroll: self.unroll,
            check_m_bounds: m % self.b_mn != 0,
            check_k_bounds: k % self.b_k != 0,
            check_n_bounds: n % self.b_mn != 0,
            coop_dim: CMMA_COOP_DIM.into(),
            num_coops: UInt::new(num_coops as u32),
            num_accumulators: UInt::new(self.alpha as u32),
            write_out_reuse_smem: self.write_out_strategy == WriteOutStrategy::ReuseSmem,
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
        CubeDim {
            x: CMMA_COOP_DIM as u32,
            y: ((self.b_mn * self.b_k) / (CMMA_TILE_SIZE * CMMA_TILE_SIZE)) as u32,
            z: 1,
        }
    }

    pub(crate) fn available_vectorizations(&self) -> Vec<u8> {
        let vectorizations = vec![8, 4, 2];
        for v in vectorizations.iter() {
            assert!(CMMA_TILE_SIZE * CMMA_TILE_SIZE % (*v as usize * CMMA_COOP_DIM) == 0);
        }
        vectorizations
    }
}

impl Init for ComptimeCmmaInfo {
    fn init(self, _context: &mut CubeContext) -> Self {
        self
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
/// Tiling 2D parameters
pub struct ComptimeCmmaInfo {
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
    pub num_coops: UInt,
    /// Number of cmma per subcube performed in one pass
    pub num_accumulators: UInt,
    /// Write out strategy: false = large, true = reuse
    pub write_out_reuse_smem: bool,
}
