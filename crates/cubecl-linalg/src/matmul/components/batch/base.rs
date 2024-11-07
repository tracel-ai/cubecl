use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::batch;
use crate::matmul::components::{
    config::MatmulConfig, global, Ident, MatmulKernel, MatmulLaunch, StageDim,
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
/// It is therefore important to use an underlying global matmul that performs check bounds,
/// and to not launch more Cubes than necessary.
pub trait Matmul<EG: Numeric>:
    'static + Send + Sync + MatmulKernel<EG, EG, Config: Config> + MatmulLaunch<EG, EG>
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
pub trait Config: MatmulConfig {
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

#[cube(launch_unchecked)]
// TODO input as references
pub(crate) fn launch<EG: Numeric, BMM: batch::Matmul<EG>>(
    lhs: Tensor<Line<EG>>,
    rhs: Tensor<Line<EG>>,
    out: Tensor<Line<EG>>,
    #[comptime] config: BMM::Config,
) {
    BMM::execute(lhs, rhs, out, config);
}
