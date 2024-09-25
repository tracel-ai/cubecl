use cubecl_core::prelude::*;

use crate::matmul::cmma::config::TileDimension;

use super::{
    strategy::{
        ComputeLoopOrderStrategy, MainLoopStrategy, RasterizationStrategy, SmemLoaderStrategy,
        WriteOutStrategy,
    },
    TileDimensionStrategy, TilingOrderStrategy,
};

pub(crate) const CMMA_PLANE_DIM: u8 = 32;

pub struct CmmaConfig {
    /// Corresponds to the number of tiles in the m dimension for a block
    pub b_m: usize,
    /// Corresponds to the number of tiles in the k dimension for a block
    pub b_k: usize,
    /// Corresponds to the number of tiles in the n dimension for a block
    pub b_n: usize,
    pub t_m: u8,
    pub t_k: u8,
    pub t_n: u8,
    /// Whether to unroll loop over k within the shared memory
    pub unroll: bool,
    pub num_compute_planes: u32,
    pub num_accumulators: u32,
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
    pub tile_dimension_strategy: TileDimensionStrategy,
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
            MainLoopStrategy::Standard,
            TileDimensionStrategy::M16K16N16,
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
        tile_dimension_strategy: TileDimensionStrategy,
    ) -> CmmaConfig {
        let tile_dim: TileDimension = tile_dimension_strategy.into();
        let t_m = tile_dim.m;
        let t_k = tile_dim.k;
        let t_n = tile_dim.n;

        assert!(b_m % tile_dim.m as usize == 0);
        assert!(b_k % tile_dim.k as usize == 0);
        assert!(b_n % tile_dim.n as usize == 0);

        let num_compute_planes = b_m as u32 / tile_dim.m as u32;
        // let num_buffers = self.b_k as u32 / tile_dim.k as u32;
        let num_accumulators = b_n as u32 / tile_dim.n as u32;

        CmmaConfig {
            b_m,
            b_k,
            b_n,
            t_m,
            t_k,
            t_n,
            unroll,
            num_compute_planes,
            num_accumulators,
            write_out_strategy,
            rasterization_strategy,
            compute_loop_order_strategy,
            lhs_smem_loader_strategy,
            rhs_smem_loader_strategy,
            main_loop_strategy,
            tile_dimension_strategy,
        }
    }

    pub(crate) fn comptime_info(&self, m: usize, k: usize, n: usize) -> ComptimeCmmaInfo {
        ComptimeCmmaInfo {
            block_size_m: self.b_m as u32,
            block_size_k: self.b_k as u32,
            block_size_n: self.b_n as u32,
            tile_size_m: self.t_m as u32,
            tile_size_k: self.t_k as u32,
            tile_size_n: self.t_n as u32,
            unroll: self.unroll,
            check_m_bounds: m % self.b_m != 0,
            check_k_bounds: k % self.b_k != 0,
            check_n_bounds: n % self.b_n != 0,
            plane_dim: CMMA_PLANE_DIM as u32,
            num_compute_planes: self.num_compute_planes,
            num_accumulators: self.num_accumulators,
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
        let y_size = self.num_compute_planes
            + match self.main_loop_strategy {
                MainLoopStrategy::Standard => 0,
                MainLoopStrategy::Split(num_load) => num_load,
            };

        CubeDim {
            x: CMMA_PLANE_DIM as u32,
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
    pub plane_dim: u32,
    /// The number of collaboration planes for compute
    pub num_compute_planes: u32,
    /// Number of cmma per subcube performed in one pass
    pub num_accumulators: u32,
    pub write_out_strategy: WriteOutStrategy,
    pub rasterization_strategy: RasterizationStrategy,
    pub compute_loop_order_strategy: ComputeLoopOrderStrategy,
    pub lhs_smem_loader_strategy: SmemLoaderStrategy,
    pub rhs_smem_loader_strategy: SmemLoaderStrategy,
    pub main_loop_strategy: MainLoopStrategy,
}
