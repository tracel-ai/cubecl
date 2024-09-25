use cubecl_core::prelude::*;

use super::{
    strategy::{
        ComputeLoopOrderStrategy, MainLoopStrategy, RasterizationStrategy, SmemLoaderStrategy,
        WriteOutStrategy,
    },
    TilingOrderStrategy,
};

// It is assumed that CMMA uses 32 units to compute 16x16x16 tiles
// TODO add number of smem banks
pub(crate) const CMMA_COOP_DIM: usize = 32;
pub(crate) const TILE_SIZE_M: usize = 16;
pub(crate) const TILE_SIZE_K: usize = 16;
pub(crate) const TILE_SIZE_N: usize = 16;

pub struct CmmaConfig {
    /// Corresponds to the number of tiles in the m dimension for a block
    pub b_m: usize,
    /// Corresponds to the number of tiles in the k dimension for a block
    pub b_k: usize,
    /// Corresponds to the number of tiles in the n dimension for a block
    pub b_n: usize,
    /// Whether to unroll loop over k within the shared memory
    pub unroll: bool,
    /// Whether to write all accumulators in different spots of a large shared memory or reuse the space
    pub write_out_strategy: WriteOutStrategy,
    /// Order in which to dispatch cubes
    pub rasterization_strategy: RasterizationStrategy,
    /// Whether to iterate on buffers or accumulators first
    pub compute_loop_order_strategy: ComputeLoopOrderStrategy,
    /// How to load and read from LHS shared memory
    pub lhs_smem_loader_strategy: SmemLoaderStrategy,
    /// How to load and read from RHS shared memory
    pub rhs_smem_loader_strategy: SmemLoaderStrategy,
    /// How to parallelize the outer loop among different warps
    pub main_loop_strategy: MainLoopStrategy,
}

impl Default for CmmaConfig {
    fn default() -> Self {
        Self::new(
            64,
            32,
            64,
            false,
            WriteOutStrategy::ReuseSmem,
            RasterizationStrategy::Swizzle,
            ComputeLoopOrderStrategy::AllAccumulatorsFirst(true),
            SmemLoaderStrategy::Tilewise(TilingOrderStrategy::RowMajor),
            SmemLoaderStrategy::Tilewise(TilingOrderStrategy::ColMajor),
            MainLoopStrategy::Standard(8),
        )
    }
}

impl CmmaConfig {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        b_m: usize,
        b_k: usize,
        b_n: usize,
        unroll: bool,
        write_out_strategy: WriteOutStrategy,
        rasterization_strategy: RasterizationStrategy,
        compute_loop_order_strategy: ComputeLoopOrderStrategy,
        lhs_smem_loader_strategy: SmemLoaderStrategy,
        rhs_smem_loader_strategy: SmemLoaderStrategy,
        main_loop_strategy: MainLoopStrategy,
    ) -> CmmaConfig {
        assert!(b_m % TILE_SIZE_M == 0);
        assert!(b_k % TILE_SIZE_K == 0);
        assert!(b_n % TILE_SIZE_N == 0);

        CmmaConfig {
            b_m,
            b_k,
            b_n,
            unroll,
            write_out_strategy,
            rasterization_strategy,
            compute_loop_order_strategy,
            lhs_smem_loader_strategy,
            rhs_smem_loader_strategy,
            main_loop_strategy,
        }
    }

    pub(crate) fn comptime_info(&self, m: usize, k: usize, n: usize) -> ComptimeCmmaInfo {
        let (num_compute_coops, num_load_coops) = match self.main_loop_strategy {
            MainLoopStrategy::Standard(num) => (num, num),
            MainLoopStrategy::Split(num_compute, num_load) => (num_compute, num_load),
        };

        ComptimeCmmaInfo {
            block_size_m: self.b_m as u32,
            block_size_k: self.b_k as u32,
            block_size_n: self.b_n as u32,
            tile_size_m: TILE_SIZE_M as u32,
            tile_size_k: TILE_SIZE_K as u32,
            tile_size_n: TILE_SIZE_N as u32,
            unroll: self.unroll,
            check_m_bounds: m % self.b_m != 0,
            check_k_bounds: k % self.b_k != 0,
            check_n_bounds: n % self.b_n != 0,
            coop_dim: CMMA_COOP_DIM as u32,
            num_compute_coops,
            num_load_coops,
            num_accumulators: (self.b_m * self.b_n / (TILE_SIZE_M * TILE_SIZE_N)) as u32
                / num_compute_coops,
            write_out_strategy: self.write_out_strategy,
            rasterization_strategy: self.rasterization_strategy,
            compute_loop_order_strategy: self.compute_loop_order_strategy,
            lhs_smem_loader_strategy: self.lhs_smem_loader_strategy,
            rhs_smem_loader_strategy: self.rhs_smem_loader_strategy,
            main_loop_strategy: self.main_loop_strategy,
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
            .rasterization_strategy
            .get_cube_dim(num_rows, num_cols, self.b_m, self.b_n);

        let mut num_iter = 1;
        for shape in output_shape.iter().take(rank - 2) {
            num_iter *= shape;
        }

        CubeCount::Static(cubes_x, cubes_y, num_iter as u32)
    }

    pub(crate) fn cube_dim(&self) -> CubeDim {
        let y_size = match self.main_loop_strategy {
            MainLoopStrategy::Standard(num) => num,
            MainLoopStrategy::Split(num_compute, num_load) => num_compute + num_load,
        };
        CubeDim {
            x: CMMA_COOP_DIM as u32,
            y: y_size,
            z: 1,
        }
    }

    pub(crate) fn available_vectorizations(&self) -> Vec<u8> {
        vec![8, 4, 2]
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
    /// Tile size along dimension m. Should correspond to cmma supported tile size
    pub tile_size_m: u32,
    /// Tile size along dimension k. Should correspond to cmma supported tile size
    pub tile_size_k: u32,
    /// Tile size along dimension n. Should correspond to cmma supported tile size
    pub tile_size_n: u32,
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
    pub write_out_strategy: WriteOutStrategy,
    pub rasterization_strategy: RasterizationStrategy,
    pub compute_loop_order_strategy: ComputeLoopOrderStrategy,
    pub lhs_smem_loader_strategy: SmemLoaderStrategy,
    pub rhs_smem_loader_strategy: SmemLoaderStrategy,
    pub main_loop_strategy: MainLoopStrategy,
}
