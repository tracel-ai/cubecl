use crate::matmul::components::stage;
use crate::matmul::components::MatrixLayout;

/// Configs that may impact performance
pub struct AdvancedConfig {
    /// Layout in which tiles should be in lhs shared memory
    pub lhs_tiling_layout: stage::TilingLayout,
    /// Layout in which tiles should be in rhs shared memory
    pub rhs_tiling_layout: stage::TilingLayout,
    /// Ensure the inputs to tile matmul are in specified layout
    ///
    /// # Notes
    ///
    /// First item is for LHS, second item is for RHS
    /// If None, the layout will be the same as in global memory
    /// If enforced layout is different from global memory,
    /// transpose will be done at loading from global memory to stage,
    /// and stage will not be vectorized.
    pub enforced_matrix_layout: (Option<MatrixLayout>, Option<MatrixLayout>),
}

impl Default for AdvancedConfig {
    fn default() -> Self {
        Self {
            lhs_tiling_layout: stage::TilingLayout::Contiguous(stage::TilingOrder::RowMajor),
            rhs_tiling_layout: stage::TilingLayout::Contiguous(stage::TilingOrder::RowMajor),
            enforced_matrix_layout: (None, None),
        }
    }
}
