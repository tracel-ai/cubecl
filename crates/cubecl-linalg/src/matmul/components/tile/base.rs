use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::{
    Ident, InputIdent, MatmulConfigFactory, MatmulSize, MatrixLayout, config::MatmulConfig,
};

pub trait TileMatmulFamily:
    MatmulConfigFactory<Input = MatmulSize, Config: TileConfig>
{
    fn tile_shape(config: &Self::Config) -> Self::Input;
    fn requires_tensor_cores() -> bool;

    type Matmul<I: Numeric, O: Numeric>: TileMatmul<I, O, Config = Self::Config>;
}

/// Provides matrix multiplication operations at the tile level.
///
/// At the tile level,
///  - Inputs are raw slices of data, called tiles.
///  - units within one plane can collaborate to solve the problem
///  - dimensions M, N and K are fixed to an integer, and the
///    matrix multiplication works only for size (M, K) · (K, N) = (M, N).
///
/// Assumptions:
///  - Slices given as inputs must always be valid. If the actual matrix multiplication
///    should be done on smaller sizes than M, N and K, padding with zeros must be done beforehand.
///  - Enough units are present to perform the whole computation
#[cube]
pub trait TileMatmul<I: Numeric, O: Numeric>: 'static + Send + Sync {
    type Config: TileConfig;
    /// Contains LHS data that can be split across the units
    type Lhs: CubeType;
    /// Contains RHS data that can be split across the units
    type Rhs: CubeType;
    /// Contains output data that can be split across the units
    type Accumulator: CubeType;

    /// Executes the matrix multiplication of LHS and RHS, adding the result to the output
    fn execute(
        lhs: &Self::Lhs,
        rhs: &Self::Rhs,
        out: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    );

    /// Create the container for LHS data
    ///
    /// # Safety
    ///
    /// This may point towards uninitialized memory.
    /// Make sure to call [fill_lhs](TileMatmul::fill_lhs) prior to [execute](TileMatmul::execute).
    fn allocate_lhs(#[comptime] config: Self::Config) -> Self::Lhs;

    /// Create the container for RHS data
    ///
    /// # Safety
    ///
    /// This may point towards uninitialized memory.
    /// Make sure to call [fill_rhs](TileMatmul::fill_lhs) prior to [execute](TileMatmul::execute).
    fn allocate_rhs(#[comptime] config: Self::Config) -> Self::Rhs;

    /// Fill the container of LHS with data
    fn fill_lhs(slice: &Tile<I>, lhs: &mut Self::Lhs, #[comptime] config: Self::Config);

    /// Fill the container of RHS with data
    fn fill_rhs(slice: &Tile<I>, rhs: &mut Self::Rhs, #[comptime] config: Self::Config);

    /// Fill the accumulator with data
    fn fill_accumulator(
        tile: &Tile<O>,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    );

    /// Write the content of the output container to the given slice
    fn read_accumulator<C: Numeric>(
        out: &Self::Accumulator,
        slice: &mut SliceMut<Line<C>>,
        #[comptime] config: Self::Config,
    );

    /// Allocate the container to receive the execution output.
    ///
    /// # Safety
    ///
    /// The output container must be initialized to some value (typically 0),
    /// because the execution adds to the already present value.
    /// Make sure to call either [fill_accumulator](TileMatmul::fill_accumulator)
    /// or [zero_accumulator](TileMatmul::zero_accumulator).
    fn allocate_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator;

    /// Fill the accumulator with zeros.
    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] config: Self::Config);
}

/// Configuration for the Tile matmul (TMM) level
pub trait TileConfig: MatmulConfig {
    /// Returns the size of the plane dimension
    fn plane_dim(&self) -> u32;

    /// Returns the [MatrixLayout] for the given ident
    fn matrix_layout(&self, ident: Ident) -> MatrixLayout;

    /// Returns the line size for the given ident
    fn line_size(&self, ident: Ident) -> u32;

    /// Returns the shape of the tiles in the three axes m, k and n.
    fn tile_shape(&self) -> &MatmulSize;
}

#[derive(CubeType)]
/// Data to be handed to the tile matmul
pub struct Tile<ES: Numeric> {
    /// Slice containing all data
    pub slice: Slice<Line<ES>>,
    /// Stride between each row/col, depending on MatrixLayout (the other is assumed to be 1)
    pub stride: u32,
}

#[cube]
impl<ES: Numeric> Tile<ES> {
    pub fn new_contiguous<T: TileConfig>(
        slice: Slice<Line<ES>>,
        #[comptime] ident: Ident,
        #[comptime] config: T,
    ) -> Tile<ES> {
        let stride = comptime! {
            (match ident.as_input() {
            InputIdent::Lhs => match config.matrix_layout(ident) {
                MatrixLayout::RowMajor => config.tile_shape().k,
                MatrixLayout::ColMajor => config.tile_shape().m,
            },
            InputIdent::Rhs => match config.matrix_layout(ident) {
                MatrixLayout::RowMajor => config.tile_shape().n,
                MatrixLayout::ColMajor => config.tile_shape().k,
            },
        }) / config.line_size(ident)};

        Tile::<ES> { slice, stride }
    }

    pub fn new_strided(slice: Slice<Line<ES>>, stride: u32) -> Tile<ES> {
        Tile::<ES> { slice, stride }
    }

    pub fn as_unlined<T: TileConfig>(
        &self,
        #[comptime] ident: Ident,
        #[comptime] config: T,
    ) -> (Slice<ES>, u32) {
        (
            self.slice.try_cast_unchecked(),
            self.stride * config.line_size(ident),
        )
    }
}
