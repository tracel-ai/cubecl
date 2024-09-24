use cubecl_core::prelude::*;

use super::strategy::{
    BlockLoopStrategy, ComputeLoopOrderStrategy, CubeDispatchStrategy, SmemLoaderStrategy,
    WriteOutStrategy,
};

// It is assumed that CMMA uses 32 units to compute 16x16x16 tiles
// TODO put it in config and split tile size into three different parameters
// TODO add number of smem banks
pub(crate) const CMMA_COOP_DIM: usize = 32;
pub(crate) const CMMA_TILE_SIZE: usize = 16;

pub struct CmmaConfig {
    /// Corresponds to the number of tiles in the m and n dimensions for a block
    pub b_mn: usize,
    /// Corresponds to the number of tiles in the k dimension for a block
    pub b_k: usize,
    /// Whether to unroll loop over k within the shared memory
    pub unroll: bool,
    /// Whether to write all accumulators in different spots of a large shared memory or reuse the space
    pub write_out_strategy: WriteOutStrategy,
    /// Order in which to dispatch cubes
    pub cube_dispatch_strategy: CubeDispatchStrategy,
    /// Whether to iterate on buffers or accumulators first
    pub compute_loop_order_strategy: ComputeLoopOrderStrategy,
    /// How to load and read from LHS shared memory
    pub lhs_smem_loader_strategy: SmemLoaderStrategy,
    /// How to load and read from RHS shared memory
    pub rhs_smem_loader_strategy: SmemLoaderStrategy,
    /// How to parallelize the outer loop among different warps
    pub block_loop_strategy: BlockLoopStrategy,
}

impl Default for CmmaConfig {
    fn default() -> Self {
        Self::new(
            128,
            16,
            false,
            WriteOutStrategy::ReuseSmem,
            CubeDispatchStrategy::Swizzle,
            ComputeLoopOrderStrategy::AllAccumulatorsFirst(true),
            SmemLoaderStrategy::TilewiseRowMajor,
            SmemLoaderStrategy::TilewiseColMajor,
            BlockLoopStrategy::Standard(8),
        )
    }
}

impl CmmaConfig {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        b_mn: usize,
        b_k: usize,
        unroll: bool,
        write_out_strategy: WriteOutStrategy,
        cube_dispatch_strategy: CubeDispatchStrategy,
        compute_loop_order_strategy: ComputeLoopOrderStrategy,
        lhs_smem_loader_strategy: SmemLoaderStrategy,
        rhs_smem_loader_strategy: SmemLoaderStrategy,
        block_loop_strategy: BlockLoopStrategy,
    ) -> CmmaConfig {
        assert!(b_mn % CMMA_TILE_SIZE == 0);
        assert!(b_k % CMMA_TILE_SIZE == 0);
        assert!(b_mn % b_k == 0);
        CmmaConfig {
            b_mn,
            b_k,
            unroll,
            write_out_strategy,
            cube_dispatch_strategy,
            compute_loop_order_strategy,
            lhs_smem_loader_strategy,
            rhs_smem_loader_strategy,
            block_loop_strategy,
        }
    }

    pub(crate) fn comptime_info(&self, m: usize, k: usize, n: usize) -> ComptimeCmmaInfo {
        let (compute_loop_order_strategy, reuse_lhs_fragment) =
            self.compute_loop_order_strategy.into();
        let (block_loop_strategy, num_compute_coops, num_load_coops) =
            self.block_loop_strategy.into();

        ComptimeCmmaInfo {
            block_size_m: self.b_mn as u32,
            block_size_k: self.b_k as u32,
            block_size_n: self.b_mn as u32,
            tile_size: CMMA_TILE_SIZE as u32,
            unroll: self.unroll,
            check_m_bounds: m % self.b_mn != 0,
            check_k_bounds: k % self.b_k != 0,
            check_n_bounds: n % self.b_mn != 0,
            coop_dim: CMMA_COOP_DIM as u32,
            num_compute_coops,
            num_load_coops,
            num_accumulators: (self.b_mn / self.b_k) as u32,
            write_out_strategy: self.write_out_strategy.into(),
            cube_dispatch_strategy: self.cube_dispatch_strategy.into(),
            compute_loop_order_strategy,
            reuse_lhs_fragment,
            lhs_smem_loader_strategy: self.lhs_smem_loader_strategy.into(),
            rhs_smem_loader_strategy: self.rhs_smem_loader_strategy.into(),
            block_loop_strategy,
        }
    }

    pub(crate) fn cube_count<R: Runtime>(
        &self,
        output_shape: &[usize],
    ) -> CubeCount<<R as Runtime>::Server> {
        let rank = output_shape.len();
        let num_rows = *output_shape.get(rank - 2).unwrap();
        let num_cols = *output_shape.get(rank - 1).unwrap();

        let (cubes_x, cubes_y) = self
            .cube_dispatch_strategy
            .get_cube_dim(num_rows, num_cols, self.b_mn);

        let mut num_iter = 1;
        for shape in output_shape.iter().take(rank - 2) {
            num_iter *= shape;
        }

        CubeCount::Static(cubes_x, cubes_y, num_iter as u32)
    }

    pub(crate) fn cube_dim(&self) -> CubeDim {
        CubeDim {
            x: CMMA_COOP_DIM as u32,
            y: self.block_loop_strategy.num_coops(),
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
pub struct ComptimeCmmaInfo {
    /// Block size along dimension of lhs
    pub block_size_m: u32,
    /// Block size along common dimension
    pub block_size_k: u32,
    /// Block size along dimension of rhs
    pub block_size_n: u32,
    /// Tile size (dimension of one side). Should correspond to cmma supported tile size
    pub tile_size: u32,
    /// Bounds must be checked on lhs dimension
    pub check_m_bounds: bool,
    /// Bounds must be checked on common dimension
    pub check_k_bounds: bool,
    /// Bounds must be checked on rhs dimension
    pub check_n_bounds: bool,
    /// Unroll
    pub unroll: bool,
    /// The number of units that can collaborate
    pub coop_dim: u32,
    /// The number of collaboration groups for compute
    pub num_compute_coops: u32,
    /// The number of collaboration groups for loading
    pub num_load_coops: u32,
    /// Number of cmma per subcube performed in one pass
    pub num_accumulators: u32,
    /// 0 = large, 1 = reuse
    pub write_out_strategy: u32,
    /// 0 = RowMajor, 1 = ColMajor, 2 = Swizzle
    pub cube_dispatch_strategy: u32,
    /// 0 = all buffers first, 1 = all accumulators first
    pub compute_loop_order_strategy: u32,
    /// Whether to reuse lhs fragment (true) or to reload it (false)
    /// Available only with all accumulators first compute loop order
    pub reuse_lhs_fragment: bool,
    /// 0 = tilewise row major, 1 = tilewise col major
    /// 2 = continous row major, 3 = continuous col major
    pub lhs_smem_loader_strategy: u32,
    /// 0 = tilewise row major, 1 = tilewise col major
    /// 2 = continous row major, 3 = continuous col major
    pub rhs_smem_loader_strategy: u32,
    /// 0 same role, 1 = split roles halfway
    pub block_loop_strategy: u32,
}
