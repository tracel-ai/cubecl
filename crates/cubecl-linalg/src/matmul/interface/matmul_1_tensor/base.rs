use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::interface::matmul_2_flow::base::FlowMatmul;

#[cube]
/// Handles the matrix multiplication of two tensors of
/// arbitrary matching shapes
/// Computes A·B where
/// - A has shape [b_1, b_2, ..., b_h, m, k]
/// - B has shape [b_1, b_2, ..., b_h, k, n]
///
/// Responsibilities:
/// - Dispatch Cubes
/// - Load balancing across SMs
/// - L2 cache reuse
/// - Matrix layout (call flow differently for transposed)
pub trait TensorMatmul {
    type FlowMatmul: FlowMatmul;

    // Tensor, with shape, stride and handle
    type Input: CubeType;
    // Tensor
    type Output: CubeType;

    fn execute(lhs: &Self::Input, rhs: &Self::Input, out: &mut Self::Output);
}
