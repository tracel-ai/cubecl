use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::block_info::BlockInfos;
use super::matrix_layout::MatrixLayout;
use super::problem::{MatmulProblem, Requirements};
use super::tile_io::{BlockReader, Loader, TileWriter};

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
/// Execute a matmul over a block, accumulating for arbitrary k-dim, using one Cube.
pub trait CubeMatmul<E: Numeric, Lhs: Loader<E>, Rhs: Loader<E>, Out: TileWriter<Line<E>>>:
    'static + Send + Sync + TensorMatmul<E>
{
    fn execute(lhs_reader: Lhs, rhs_reader: Rhs, out_writer: Out, k_range: (u32, u32));
}

#[cube]
/// Execute a matmul over a fixed-size block, using one Cube.
pub trait BlockMatmul<
    E: Numeric,
    Lhs: BlockReader<E>,
    Rhs: BlockReader<E>,
    Out: TileWriter<Line<E>>,
>: 'static + Send + Sync + FixedShapeMatmul<E, E>
{
    type Config;
    type Accumulator: CubeType;

    fn execute(lhs: &Lhs, rhs: &Rhs, acc: &mut Self::Accumulator);

    fn acc_init_zeros() -> Self::Accumulator;
    fn acc_read(acc: &Self::Accumulator, out: &mut Out);
}

pub trait Matmul<I: Numeric, O: Numeric> {
    fn can_process(problem: MatmulProblem) -> bool;
    fn requirements(problem: MatmulProblem) -> Requirements;

    fn block_infos() -> BlockInfos;
}

pub trait TensorMatmul<E: Numeric>: Matmul<E, E> {
    unsafe fn launch_unchecked<R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount<<R as Runtime>::Server>,
        lhs: TensorArg<'_, R>,
        rhs: TensorArg<'_, R>,
        out: TensorArg<'_, R>,
        layouts: (MatrixLayout, MatrixLayout),
    );
}

pub trait FixedShapeMatmul<I: Numeric, O: Numeric>: Matmul<I, O> {
    const M: u32;
    const N: u32;
    const K: u32;

    unsafe fn launch_unchecked<R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount<<R as Runtime>::Server>,
        lhs: ArrayArg<'_, R>,
        rhs: ArrayArg<'_, R>,
        out: ArrayArg<'_, R>,
        layouts: (MatrixLayout, MatrixLayout),
    );
}

#[cube]
/// Executes a small matmul using one plane
pub trait MatmulInstruction<I: Numeric, O: Numeric>:
    'static + Send + Sync + FixedShapeMatmul<I, O>
{
    type Config;
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
