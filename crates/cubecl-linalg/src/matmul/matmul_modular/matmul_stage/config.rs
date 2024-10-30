use crate::matmul::matmul_modular::cmma_matmul::stage::TilingOrderConfig;
use crate::matmul::matmul_modular::config::MatmulConfig;
use crate::matmul::matmul_modular::matmul_tile::TmmConfig;
use crate::matmul::matmul_modular::matrix::{Ident, MatrixLayout};
use crate::matmul::matmul_modular::stage_dim::StageDim;

/// Configuration for the Stage matmul (SMM) level
pub trait SmmConfig: MatmulConfig {
    /// Underlying Tile matmul config
    type TmmConfig: TmmConfig;

    /// Convert itself to the underlying tile matmul config
    fn to_tmm_config(self) -> Self::TmmConfig;

    /// Returns the line size for the given ident
    fn line_size(&self, ident: Ident) -> u32;

    /// Returns the [StageDim] for the given ident
    fn stage_dim(&self, ident: Ident) -> StageDim;

    /// Returns the [MatrixLayout] for the given ident
    fn layout(&self, ident: Ident) -> MatrixLayout;

    /// Returns the number of planes in the cube
    fn num_planes(&self) -> u32;

    /// Returns the size of the plane dimension
    fn plane_dim(&self) -> u32;

    /// Returns the order in which tiles should be loaded to the stage
    fn tiling_order(&self) -> TilingOrderConfig;
}
