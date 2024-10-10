use cubecl_core::prelude::*;

use crate::matmul::cmma::config::TileDimension;

use super::{
    strategy::{
        ComputeLoopOrderStrategy, MainLoopStrategy, RasterizationStrategy, SmemLoaderStrategy,
        WriteOutStrategy,
    },
    NumComputePlanesStrategy, TileDimensionStrategy, TilingOrderStrategy,
};

pub(crate) const CMMA_PLANE_DIM: u8 = 32;

pub struct CmmaConfig {
    /// Corresponds to the number of tiles in the m dimension for a block
    pub b_m: u32,
    /// Corresponds to the number of tiles in the k dimension for a block
    pub b_k: u32,
    /// Corresponds to the number of tiles in the n dimension for a block
    pub b_n: u32,
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
    pub tile_dimension_strategy: TileDimensionStrategy,
    pub num_compute_planes_strategy: NumComputePlanesStrategy,
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
            NumComputePlanesStrategy::NumTilesLhs,
        )
    }
}

impl CmmaConfig {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        b_m: u32,
        b_k: u32,
        b_n: u32,
        unroll: bool,
        write_out_strategy: WriteOutStrategy,
        rasterization_strategy: RasterizationStrategy,
        compute_loop_order_strategy: ComputeLoopOrderStrategy,
        lhs_smem_loader_strategy: SmemLoaderStrategy,
        rhs_smem_loader_strategy: SmemLoaderStrategy,
        main_loop_strategy: MainLoopStrategy,
        tile_dimension_strategy: TileDimensionStrategy,
        num_compute_planes_strategy: NumComputePlanesStrategy,
    ) -> CmmaConfig {
        // Don't modify things here
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
            tile_dimension_strategy,
            num_compute_planes_strategy,
        }
    }

    // TODO: bad
    fn get_num_compute_planes(&self) -> u32 {
        let tile_dim: TileDimension = self.tile_dimension_strategy.into();

        let num_tiles_m = self.b_m / tile_dim.m;
        let num_tiles_k = self.b_k / tile_dim.k;

        self.num_compute_planes_strategy
            .get_num_compute_planes(num_tiles_m, num_tiles_k)
    }

    pub(crate) fn comptime_info(&self, m: u32, k: u32, n: u32) -> ComptimeCmmaInfo {
        let tile_dim: TileDimension = self.tile_dimension_strategy.into();
        let t_m = tile_dim.m;
        let t_k = tile_dim.k;
        let t_n = tile_dim.n;

        assert!(self.b_m % t_m == 0);
        assert!(self.b_k % t_k == 0);
        assert!(self.b_n % t_n == 0);

        let num_tiles_m = self.b_m / t_m;
        let num_tiles_k = self.b_k / t_k;
        let num_tiles_n = self.b_n / t_n;

        let num_compute_planes = self
            .num_compute_planes_strategy
            .get_num_compute_planes(num_tiles_m, num_tiles_k);
        let num_accumulators = self.num_compute_planes_strategy.get_num_accumulators(
            num_tiles_m,
            num_tiles_k,
            num_tiles_n,
        );
        let num_buffers = num_tiles_k;

        let num_load_planes = self
            .main_loop_strategy
            .get_num_load_planes(num_compute_planes);

        if let SmemLoaderStrategy::Tilewise(_) = self.lhs_smem_loader_strategy {
            assert!(num_load_planes == num_tiles_m * num_tiles_k);
        }
        if let SmemLoaderStrategy::Tilewise(_) = self.rhs_smem_loader_strategy {
            assert!(num_load_planes == num_tiles_k * num_tiles_n);
        }

        assert!(
            num_tiles_k * num_tiles_n >= num_load_planes,
            "Otherwise can do out of bounds"
        );

        ComptimeCmmaInfo {
            block_size_m: self.b_m,
            block_size_k: self.b_k,
            block_size_n: self.b_n,
            tile_size_m: t_m,
            tile_size_k: t_k,
            tile_size_n: t_n,
            unroll: self.unroll,
            check_m_bounds: m % self.b_m != 0,
            check_k_bounds: k % self.b_k != 0,
            check_n_bounds: n % self.b_n != 0,
            plane_dim: CMMA_PLANE_DIM as u32,
            num_compute_planes,
            num_buffers,
            num_accumulators,
            write_out_strategy: self.write_out_strategy,
            rasterization_strategy: self.rasterization_strategy,
            compute_loop_order_strategy: self.compute_loop_order_strategy,
            lhs_smem_loader_strategy: self.lhs_smem_loader_strategy,
            rhs_smem_loader_strategy: self.rhs_smem_loader_strategy,
            main_loop_strategy: self.main_loop_strategy,
            num_compute_planes_strategy: self.num_compute_planes_strategy,
        }
    }

    pub(crate) fn cube_count(&self, output_shape: &[usize]) -> CubeCount {
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
        CubeDim {
            x: CMMA_PLANE_DIM as u32,
            y: self
                .main_loop_strategy
                .get_num_planes(self.get_num_compute_planes()),
            z: 1,
        }
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
    /// The number of buffers, should equal block_size_k / tile_size_k
    pub num_buffers: u32,
    /// Number of cmma per subcube performed in one pass
    pub num_accumulators: u32,
    pub write_out_strategy: WriteOutStrategy,
    pub rasterization_strategy: RasterizationStrategy,
    pub compute_loop_order_strategy: ComputeLoopOrderStrategy,
    pub lhs_smem_loader_strategy: SmemLoaderStrategy,
    pub rhs_smem_loader_strategy: SmemLoaderStrategy,
    pub main_loop_strategy: MainLoopStrategy,
    pub num_compute_planes_strategy: NumComputePlanesStrategy,
}
