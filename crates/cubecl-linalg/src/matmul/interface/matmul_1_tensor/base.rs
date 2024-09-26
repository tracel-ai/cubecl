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

pub trait MatmulConfig {
    fn init(cube_dim: (u32, u32, u32)) -> Self;
}

#[cube]
pub trait BatchMatmul<N: Numeric> {
    type Config;

    fn execute(
        lhs: &Tensor<Line<N>>,
        rhs: &Tensor<Line<N>>,
        out: &mut Tensor<Line<N>>,
        #[comptime] config: Self::Config,
    );
}

#[cube]
pub trait Matmul<N: Numeric> {
    type Config;

    fn execute(
        lhs: &Matrix<Line<N>>,
        rhs: &Matrix<Line<N>>,
        out: &mut MatrixMut<Line<N>>,
        #[comptime] config: Self::Config,
    );
}

#[cube]
pub trait MultiplyAddAndAccumulate<N: Numeric> {
    type Config;
    type Accumulator: CubeType;

    fn execute(
        lhs: &Matrix<Line<N>>,
        rhs: &Matrix<Line<N>>,
        acc: &mut Self::Accumulator,
        #[comptime] config: &Self::Config,
    );

    fn acc_init_zeros(#[comptime] config: &Self::Config) -> Self::Accumulator;
    fn acc_init(matrix: &Matrix<Line<N>>, #[comptime] config: &Self::Config) -> Self::Accumulator;
    fn acc_read(
        acc: &Self::Accumulator,
        out: &mut Matrix<Line<N>>,
        #[comptime] config: &Self::Config,
    ) -> Self::Accumulator;
}

#[derive(CubeType)]
pub struct Matrix<'b, N: CubePrimitive> {
    pub slice: Slice<'b, N>,
    pub strides: (u32, u32),
    pub shape: (u32, u32),
}

#[derive(CubeType)]
pub struct MatrixMut<'b, N: CubePrimitive> {
    pub slice: SliceMut<'b, N>,
    pub strides: (u32, u32),
    pub shape: (u32, u32),
}

#[cube]
pub fn allo<N: Numeric>(
    lhs: &Matrix<Line<N>>,
    rhs: &Matrix<Line<N>>,
    out: &mut MatrixMut<Line<N>>,
) {
    out.slice[0] = lhs.slice[0] * rhs.slice[0];
}
