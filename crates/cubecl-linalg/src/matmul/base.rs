use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::matrix_layout::MatrixLayout;
use super::tile_io::{TileReader, TileWriter};

#[cube]
/// Execute a matmul on a whole tensor
pub trait BatchMatmul<N: Numeric> {
    type Config;

    fn execute(
        lhs: &Tensor<Line<N>>,
        rhs: &Tensor<Line<N>>,
        out: &mut Tensor<Line<N>>,
        #[comptime] config: &Self::Config,
    );
}

#[cube]
/// Execute a matmul over matrices.
pub trait BlockMatmul<
    E: Numeric,
    Lhs: TileReader<Line<E>>,
    Rhs: TileReader<Line<E>>,
    Out: TileWriter<Line<E>>,
>: 'static + Send + Sync
{
    type Config;
    type Accumulator: CubeType;
    const M: u32;
    const N: u32;
    const K: u32;

    fn execute(
        lhs: Lhs,
        rhs: Rhs,
        acc: &mut Self::Accumulator,
        #[comptime] layouts: (MatrixLayout, MatrixLayout),
    );

    fn acc_init_zeros() -> Self::Accumulator;
    fn acc_read(acc: &Self::Accumulator, out: &mut Out);
}

#[cube]
/// Executes a matmul at the lowest level
pub trait MatmulInstruction<I: Numeric, O: Numeric>: 'static + Send + Sync {
    type Config;
    type Lhs: CubeType;
    type Rhs: CubeType;
    type Out: CubeType;
    const M: u32;
    const N: u32;
    const K: u32;

    fn execute(lhs: &Self::Lhs, rhs: &Self::Rhs, out: &mut Self::Out);

    fn init_lhs(#[comptime] layout: MatrixLayout) -> Self::Lhs;
    fn init_rhs(#[comptime] layout: MatrixLayout) -> Self::Rhs;

    fn fill_lhs<C: CubePrimitive>(slice: &Slice<'_, C>, lhs: &mut Self::Lhs);
    fn fill_rhs<C: CubePrimitive>(slice: &Slice<'_, C>, rhs: &mut Self::Rhs);

    fn init_output() -> Self::Out;
    fn read_output<C: CubePrimitive>(out: &Self::Out, slice: &mut SliceMut<'_, C>);
}
