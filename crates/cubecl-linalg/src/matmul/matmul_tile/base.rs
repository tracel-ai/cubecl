use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::Matmul;

use super::TmmConfig;

#[cube]
pub trait TileMatmul<I: Numeric, O: Numeric, T: TmmConfig>:
    'static + Send + Sync + Matmul<I, O, Config = T>
{
    const M: u32;
    const N: u32;
    const K: u32;

    type Lhs: CubeType;
    type Rhs: CubeType;
    type Out: CubeType;

    fn execute(lhs: &Self::Lhs, rhs: &Self::Rhs, out: &mut Self::Out);

    fn init_lhs(#[comptime] config: T) -> Self::Lhs;
    fn init_rhs(#[comptime] config: T) -> Self::Rhs;

    fn fill_lhs(slice: &Slice<'_, Line<I>>, lhs: &mut Self::Lhs, #[comptime] config: T);
    fn fill_rhs(slice: &Slice<'_, Line<I>>, rhs: &mut Self::Rhs, #[comptime] config: T);

    fn init_output() -> Self::Out;
    fn read_output<C: Numeric>(
        out: &Self::Out,
        slice: &mut SliceMut<'_, Line<C>>,
        #[comptime] config: T,
    );
}
