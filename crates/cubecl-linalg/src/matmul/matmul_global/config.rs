use crate::matmul::cmma_matmul::stage::TilingOrderConfig;
use crate::matmul::config::MatmulConfig;
use crate::matmul::matmul_stage::SmmConfig;
use crate::matmul::matrix::{Ident, MatrixLayout};
use crate::matmul::stage_dim::StageDim;

/// Configuration for the Global matmul (GMM) level
pub trait GmmConfig: MatmulConfig {
    /// Underlying Stage matmul config
    type SmmConfig: SmmConfig;

    /// Convert itself to the underlying stage matmul config
    fn to_smm_config(&self) -> Self::SmmConfig;

    /// Returns the line size for the given ident
    fn line_size(&self, ident: Ident) -> u32;

    /// Returns the [StageDim] for the given ident
    fn stage_dim(&self, ident: Ident) -> StageDim;

    /// Returns the [MatrixLayout] for the given ident
    fn layout(&self, ident: Ident) -> MatrixLayout;

    /// Returns the line size of the shared memory used for reorganizing the output
    /// before writing to global memory
    fn out_smem_line_size(&self) -> u32;

    /// Returns the number of planes in the cube
    fn num_planes(&self) -> u32;

    /// Returns the size of the plane dimension
    fn plane_dim(&self) -> u32;

    /// Returns the order in which tiles should be loaded to the stage
    fn tiling_order(&self) -> TilingOrderConfig;

    /// Whether it is necessary to add bound checks in the m dimension
    fn check_m_bounds(&self) -> bool;

    /// Whether it is necessary to add bound checks in the n dimension
    fn check_n_bounds(&self) -> bool;
}
