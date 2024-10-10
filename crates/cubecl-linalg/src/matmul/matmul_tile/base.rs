use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::FixedShapeMatmul;

#[cube]
/// Executes a small matmul using one plane
pub trait MatmulInstruction<I: Numeric, O: Numeric>:
    'static + Send + Sync + FixedShapeMatmul<I, O>
{
    type Lhs: CubeType;
    type Rhs: CubeType;
    type Out: CubeType;

    fn execute(lhs: &Self::Lhs, rhs: &Self::Rhs, out: &mut Self::Out);

    fn init_lhs(#[comptime] layout: MatrixLayout) -> Self::Lhs;
    fn init_rhs(#[comptime] layout: MatrixLayout) -> Self::Rhs;

    fn fill_lhs<C: CubePrimitive>(slice: &Slice<'_, C>, lhs: &mut Self::Lhs);
    fn fill_rhs<C: CubePrimitive>(slice: &Slice<'_, C>, rhs: &mut Self::Rhs);

    fn init_output() -> Self::Out;
    fn read_output<C: CubePrimitive>(out: &Self::Out, slice: &mut SliceMut<'_, C>);
}
