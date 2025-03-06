use crate::matmul::components::MatrixLayout;

/// Configs that may impact performance
#[derive(Default)]
pub struct AdvancedConfig {
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
