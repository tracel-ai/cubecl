use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::{
    config::MatmulConfig, global, matrix::Ident, stage_dim::StageDim, MatmulKernel, MatmulLaunch,
};

#[cube]
/// Provides matrix multiplication operations at the batch level.
///
/// At the batch level,
///  - Inputs are whole tensors in global memory.
///  - All Cubes can collaborate to solve the problem
///  - Dimensions M, N and K can be arbitrary large,
///    as well as the number of batches.
///
/// # Assumptions
/// - Line sizes of the inputs evenly divide the dimension they are aligned with.
/// - Enough Cubes are launched to perform the whole computation.
///
/// # Safety
///
/// It is not assumed that the matmul's dimensions match its inputs dimensions perfectly.
/// It it therefore important to use an underlying global matmul that performs check bounds,
/// and to not launch more Cubes than necessary.
pub trait Matmul<EG: Numeric, B: BmmConfig>:
    'static + Send + Sync + MatmulKernel<EG, EG, Config = B> + MatmulLaunch<EG, EG>
{
    /// Performs batchwise matrix multiplication over tensors.
    fn execute(
        lhs: Tensor<Line<EG>>,
        rhs: Tensor<Line<EG>>,
        out: Tensor<Line<EG>>,
        #[comptime] config: Self::Config,
    );
}

/// Configuration for the Batch matmul (BMM) level
pub trait BmmConfig: MatmulConfig {
    /// Underlying Global matmul config
    type GmmConfig: global::Config;

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
