use cubecl_core as cubecl;
use cubecl_core::prelude::*;

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
pub trait Matmul<E: Numeric> {
    type Config;
    type Accumulator: CubeType;

    fn execute(
        lhs: &Matrix<Line<E>>,
        rhs: &Matrix<Line<E>>,
        acc: &mut Self::Accumulator,
        #[comptime] definition: MatmulDefinition,
        #[comptime] config: &Self::Config,
    );

    fn acc_init_zeros(#[comptime] config: &Self::Config) -> Self::Accumulator;
    // fn acc_init(matrix: &Matrix<Line<E>>, #[comptime] config: &Self::Config) -> Self::Accumulator;
    fn acc_read(
        acc: &Self::Accumulator,
        out: &mut MatrixMut<Line<E>>,
        #[comptime] config: &Self::Config,
    );
}

pub struct MatmulDefinition {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

#[cube]
/// Executes a matmul at the lowest level
pub trait MatmulInstruction<I: Numeric, O: Numeric, const M: u8, const N: u8, const K: u8> {
    type Config;
    type Input: CubeType;
    type Output: CubeType;

    fn execute(lhs: &Self::Input, rhs: &Self::Input, out: &mut Self::Output);

    fn fill_input<C: CubePrimitive>(slice: &Slice<'_, C>) -> Self::Input;
    fn init_output() -> Self::Output;
    fn read_output<C: CubePrimitive>(out: &Self::Output, slice: &mut SliceMut<'_, C>);
    fn slice_length() -> u32;
}

#[derive(CubeType)]
pub struct Matrix<'b, N: CubePrimitive> {
    pub slice: Slice<'b, N>,
    pub layout: MatrixLayout,
}

#[derive(CubeType)]
pub struct MatrixMut<'b, N: CubePrimitive> {
    pub slice: SliceMut<'b, N>,
    pub layout: MatrixLayout,
}

#[derive(CubeType, Copy, Clone)]
pub enum MatrixLayout {
    Row,
    Col,
}

// #[cube]
// pub fn as_cmma_layout(#[comptime] layout: MatrixLayout) -> cmma::MatrixLayout {
//     match layout {
//         MatrixLayout::Row => cmma::MatrixLayout::RowMajor,
//         MatrixLayout::Col => cmma::MatrixLayout::ColMajor,
//     }
// }
