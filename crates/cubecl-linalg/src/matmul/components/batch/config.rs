use crate::matmul::components::config::MatmulConfig;
use crate::matmul::components::global::GmmConfig;
use crate::matmul::components::matrix::Ident;
use crate::matmul::components::stage_dim::StageDim;

/// Configuration for the Batch matmul (BMM) level
pub trait BmmConfig: MatmulConfig {
    /// Underlying Global matmul config
    type GmmConfig: GmmConfig;

    /// Convert itself to the underlying global matmul config
    fn to_gmm_config(&self) -> Self::GmmConfig;

    /// Returns the [StageDim] for the given ident
    fn stage_dim(&self, ident: Ident) -> StageDim;

    /// Returns the number of cubes launched across the x dimension
    fn cube_count_x(&self) -> u32;
    /// Returns the number of cubes launched across the y dimension
    fn cube_count_y(&self) -> u32;

    /// Returns the largest m dimension supported with these configs
    fn max_m(&self) -> u32;
    /// Returns the largest n dimension supported with these configs
    fn max_n(&self) -> u32;
    /// Returns the largest number of batches supported with these configs
    fn max_batches(&self) -> u32;
}
