use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::{
    Ident, InputIdent, MatmulConfigFactory, MatmulPrecision, MatrixLayout, TileSize,
    config::MatmulConfig, stage::StageVectorization,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct TileMatmulConfigInput {
    pub vectorization: StageVectorization,
    pub tile_size: TileSize,
}

pub trait TileMatmulFamily:
    MatmulConfigFactory<Input = TileMatmulConfigInput, Config: TileConfig>
{
    fn requires_tensor_cores() -> bool;

    type Matmul<MP: MatmulPrecision>: TileMatmul<MP, Config = Self::Config>;
}

/// Provides matrix multiplication operations at the tile level.
///
/// At the tile level,
///  - Dimensions M, N and K are fixed to an integer, and the
///    matrix multiplication works only for size (M, K) · (K, N) = (M, N).
///
/// Assumptions:
///  - Inputs must always be valid. If the actual matrix multiplication
///    should be done on smaller sizes than M, N and K, padding with zeros must be done beforehand.
///  - Enough units are present to perform the whole computation
#[cube]
pub trait TileMatmul<MP: MatmulPrecision>: 'static + Send + Sync {
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

    /// Fill the container of LHS with data
    fn fill_lhs(slice: &Tile<MP::ES>, lhs: &mut Self::Lhs, #[comptime] config: Self::Config);

    /// Create the container for RHS data
    ///
    /// # Safety
    ///
    /// This may point towards uninitialized memory.
    /// Make sure to call [fill_rhs](TileMatmul::fill_lhs) prior to [execute](TileMatmul::execute).
    fn allocate_rhs(#[comptime] config: Self::Config) -> Self::Rhs;

    /// Fill the container of RHS with data
    fn fill_rhs(slice: &Tile<MP::ES>, rhs: &mut Self::Rhs, #[comptime] config: Self::Config);

    /// Allocate the container to receive the execution output.
    ///
    /// # Safety
    ///
    /// The output container must be initialized to some value (typically 0),
    /// because the execution adds to the already present value.
    /// Make sure to call either [fill_accumulator](TileMatmul::fill_accumulator)
    /// or [zero_accumulator](TileMatmul::zero_accumulator).
    fn allocate_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator;

    /// Fill the accumulator with data
    fn fill_accumulator(
        tile: &Tile<MP::EA>,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    );

    /// Fill the accumulator with zeros.
    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] config: Self::Config);

    /// Write the content of the output container to the given slice
    fn write_results(
        out: &Self::Accumulator,
        slice: &mut SliceMut<Line<MP::EO>>,
        #[comptime] config: Self::Config,
    );
}

/// Configuration for the Tile matmul (TMM) level
pub trait TileConfig: MatmulConfig {
    /// Returns the size of the plane dimension
    fn plane_dim(&self) -> u32;

    /// Returns the [MatrixLayout] for the given ident
    fn matrix_layout(&self, ident: Ident) -> MatrixLayout;

    /// Returns the line size for the given ident
    fn stage_line_size(&self, ident: Ident) -> u32;

    /// Returns the shape of the tiles in the three axes m, k and n.
    fn tile_size(&self) -> &TileSize;
}

#[derive(CubeType, Clone)]
/// Data to be handed to the tile matmul
pub struct Tile<ES: Numeric> {
    /// Slice containing all data
    pub slice: Slice<Line<ES>>,
    /// Stride between each row/col, depending on MatrixLayout (the other is assumed to be 1)
    pub stride: u32,
    #[cube(comptime)]
    pub layout: MatrixLayout,
}

#[cube]
impl<ES: Numeric> Tile<ES> {
    pub fn new_contiguous<T: TileConfig>(
        slice: Slice<Line<ES>>,
        #[comptime] ident: Ident,
        #[comptime] config: T,
    ) -> Tile<ES> {
        let layout = config.matrix_layout(ident);
        let stride = comptime! {
            (match ident.as_input_ident() {
            InputIdent::Lhs => match layout {
                MatrixLayout::RowMajor => config.tile_size().k(),
                MatrixLayout::ColMajor => config.tile_size().m(),
            },
            InputIdent::Rhs => match layout {
                MatrixLayout::RowMajor => config.tile_size().n(),
                MatrixLayout::ColMajor => config.tile_size().k(),
            },
        }) / config.stage_line_size(ident)};

        Tile::<ES> {
            slice,
            stride,
            layout,
        }
    }

    pub fn new_strided(
        slice: Slice<Line<ES>>,
        stride: u32,
        #[comptime] layout: MatrixLayout,
    ) -> Tile<ES> {
        Tile::<ES> {
            slice,
            stride,
            layout,
        }
    }

    pub fn as_unlined<T: TileConfig>(
        &self,
        #[comptime] ident: Ident,
        #[comptime] config: T,
    ) -> (Slice<ES>, u32) {
        (
            self.slice.try_cast_unchecked(),
            self.stride * config.stage_line_size(ident),
        )
    }

    pub fn get_line(&self, strided: u32, contiguous: u32) -> Line<ES> {
        self.slice[strided * self.stride + contiguous]
    }

    pub fn get_segment_as_slice(&self, index: u32, #[comptime] num_lines: u32) -> Slice<Line<ES>> {
        let start = index * self.stride;
        self.slice.slice(start, start + num_lines)
    }
}
