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
pub trait Matmul<E: Numeric, Lhs: TileReader<Line<E>>, Rhs: TileReader<Line<E>>> {
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

#[cube]
/// Executes a matmul at the lowest level
pub trait MatmulInstruction<I: Numeric, O: Numeric> {
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

#[cube]
/// Defines the number of tiles and their size in each plane
pub trait TileReader<E: CubeType>: CubeType {
    const NUM_TILES_X: u32;
    const NUM_TILES_Y: u32;

    const TILE_SIZE_X: u32;
    const TILE_SIZE_Y: u32;

    fn read(reader: &Self, pos_x: u32, pos_y: u32) -> Slice<'_, E>;
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

#[derive(Copy, Clone)]
pub enum MatrixLayout {
    Row,
    Col,
}

impl CubeType for MatrixLayout {
    type ExpandType = Self;
}

impl Init for MatrixLayout {
    fn init(self, _context: &mut CubeContext) -> Self {
        self
    }
}

impl IntoRuntime for MatrixLayout {
    fn __expand_runtime_method(self, _context: &mut CubeContext) -> Self::ExpandType {
        self
    }
}

#[cube]
pub fn as_cmma_layout(#[comptime] layout: MatrixLayout) -> cmma::MatrixLayout {
    match layout {
        MatrixLayout::Row => cmma::MatrixLayout::RowMajor,
        MatrixLayout::Col => cmma::MatrixLayout::ColMajor,
    }
}
