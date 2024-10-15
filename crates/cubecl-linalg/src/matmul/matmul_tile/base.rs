use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::FixedShapeMatmul;

#[cube]
/// Executes a small matmul using one plane
pub trait TileMatmul<I: Numeric, O: Numeric>:
    'static + Send + Sync + FixedShapeMatmul<I, O>
{
    type Lhs: CubeType;
    type Rhs: CubeType;
    type Out: CubeType;

    fn execute(lhs: &Self::Lhs, rhs: &Self::Rhs, out: &mut Self::Out);

    fn init_lhs(#[comptime] layout: MatrixLayout) -> Self::Lhs;
    fn init_rhs(#[comptime] layout: MatrixLayout) -> Self::Rhs;

    fn fill_lhs(slice: &Slice<'_, Line<I>>, lhs: &mut Self::Lhs);
    fn fill_rhs(slice: &Slice<'_, Line<I>>, rhs: &mut Self::Rhs);

    fn init_output() -> Self::Out;
    fn read_output<C: Numeric>(out: &Self::Out, slice: &mut SliceMut<'_, Line<C>>);
}
