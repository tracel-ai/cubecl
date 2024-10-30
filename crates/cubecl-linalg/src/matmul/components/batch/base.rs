use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::{Matmul, MatmulLaunch};

use super::BmmConfig;

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
pub trait BatchMatmul<EG: Numeric, B: BmmConfig>:
    'static + Send + Sync + Matmul<EG, EG, Config = B> + MatmulLaunch<EG, EG>
{
    /// Performs batchwise matrix multiplication over tensors.
    fn execute(
        lhs: Tensor<Line<EG>>,
        rhs: Tensor<Line<EG>>,
        out: Tensor<Line<EG>>,
        #[comptime] config: Self::Config,
    );
}
