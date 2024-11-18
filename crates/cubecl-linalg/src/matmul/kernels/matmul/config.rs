use crate::matmul::components::stage;
use crate::matmul::components::Ident;
use crate::matmul::components::MatrixLayout;
use crate::matmul::components::StageDim;

/// Configs that may impact performance
pub struct AdvancedConfig {
    /// Order in which tiles should be in lhs shared memory
    pub lhs_tiling_order: stage::TilingOrderConfig,
    /// Order in which tiles should be in rhs shared memory
    pub rhs_tiling_order: stage::TilingOrderConfig,
    /// Ensure the inputs to tile matmul are in specified layout
    ///
    /// # Notes
    ///
    /// First item is for LHS, second item is for RHS
    /// If None, the layout will be the same as in global memory
    /// If enforced layout is different from global memory,
    /// transpose will be done at loading from global memory to stage,
    /// and stage will not be vectorized.
    pub enforced_tile_layout: (Option<MatrixLayout>, Option<MatrixLayout>),
}

impl Default for AdvancedConfig {
    fn default() -> Self {
        Self {
            lhs_tiling_order: stage::TilingOrderConfig::RowMajor,
            rhs_tiling_order: stage::TilingOrderConfig::RowMajor,
            enforced_tile_layout: (None, None),
        }
    }
}

pub fn create_stage_dim(
    stage_m: u32,
    stage_n: u32,
    stage_k: u32,
    tile_m: u32,
    tile_n: u32,
    tile_k: u32,
) -> (StageDim, StageDim, StageDim) {
    let lhs_stage_dim = StageDim::new(
        Ident::Lhs,
        tile_m,
        tile_k,
        stage_m / tile_m,
        stage_k / tile_k,
    );

    let rhs_stage_dim = StageDim::new(
        Ident::Rhs,
        tile_k,
        tile_n,
        stage_k / tile_k,
        stage_n / tile_n,
    );

    let out_stage_dim = StageDim::new(
        Ident::Out,
        tile_m,
        tile_n,
        stage_m / tile_m,
        stage_n / tile_n,
    );

    (lhs_stage_dim, rhs_stage_dim, out_stage_dim)
}
