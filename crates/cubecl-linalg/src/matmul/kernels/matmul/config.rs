use crate::matmul::components::stage;
use crate::matmul::components::MatrixLayout;

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
