use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::CubeOption;

use crate::matmul::components::{
    Ident, MatmulConfigFactory, MatmulPrecision, MatmulSize, MatrixLayout, TilingDimensions,
    config::MatmulConfig,
    global::{self, AccumulatorLoader, IndexedQuantization},
    tile::TileConfig,
};

use super::{StageEventListener, TilingLayout};

pub trait ReaderFamily {
    type Reader<I: Numeric, T: TilingLayout>: CubeType;
}

pub trait StageMatmulFamily:
    MatmulConfigFactory<Config: StageConfig> + Send + Sync + 'static
{
    type LhsReader: ReaderFamily;
    type RhsReader: ReaderFamily;

    /// Returns the shape of the stage. This is the number of elements per axis.
    fn stage_shape(config: &Self::Config) -> MatmulSize;

    /// Returns the number of tiles in each axis of the stage.
    fn tile_count(config: &Self::Config) -> MatmulSize;

    type Matmul<MP: MatmulPrecision, TL: TilingLayout, TR: TilingLayout>: StageMatmul<
            MP,
            Config = Self::Config,
            LhsReader = <Self::LhsReader as ReaderFamily>::Reader<MP::ES, TL>,
            RhsReader = <Self::RhsReader as ReaderFamily>::Reader<MP::ES, TR>,
        >;
}

#[cube]
/// Provides matrix multiplication operations at the stage level.
///
/// At the stage level,
///  - Inputs are staged into an intermediate memory called stage (typically a shared memory).
///  - All planes within a Cube can collaborate to solve the problem
///  - Dimensions M, N and K are fixed to an integer, and the
///    matrix multiplication works only for size (M, K) · (K, N) = (M, N).
///    These integers are multiples of the underlying Tile matmul,
///    corresponding to the number of tiles in each dimension.
///
/// Assumptions:
///  - Data given as inputs by stage readers must always be valid. If the actual matrix multiplication
///    should be done on smaller sizes than M, N and K, padding with zeros must be done beforehand.
///  - Enough planes are launched to perform the whole computation
pub trait StageMatmul<MP: MatmulPrecision>: 'static + Send + Sync {
    type Config: StageConfig;

    /// Contains the matrix multiplication output, that can be shared across the different planes of the cube.
    /// The same Accumulator will be added to across multiple executions of the stage matmul.
    type Accumulator: CubeType;

    type LhsReader: CubeType;
    type RhsReader: CubeType;

    type LhsTile: CubeType;
    type RhsTile: CubeType;

    /// Executes the matrix multiplication of LHS and RHS, adding the result to the accumulator
    ///
    /// Equivalent to execute_with_listener with SEL:=NoEvent
    ///
    /// # Quantization
    ///
    /// If scaling is provided, the matmul will be performed in a quantized version.
    /// This assumes that [read_accumulator] is called with some `quantization` provided.
    fn execute(
        lhs: &Self::LhsReader,
        rhs: &Self::RhsReader,
        instruction_lhs: &mut Self::LhsTile,
        instruction_rhs: &mut Self::RhsTile,
        acc: &mut Self::Accumulator,
        scaling: CubeOption<f32>,
        #[comptime] config: Self::Config,
    );

    /// Executes the matrix multiplication of LHS and RHS, with the addition of injected
    /// [event listener](StageEventListener).
    fn execute_with_listener<SEL: StageEventListener>(
        lhs: &Self::LhsReader,
        rhs: &Self::RhsReader,
        instruction_lhs: &mut Self::LhsTile,
        instruction_rhs: &mut Self::RhsTile,
        acc: &mut Self::Accumulator,
        scaling: CubeOption<f32>,
        #[comptime] config: Self::Config,
        listener: SEL,
    );

    fn init_tile_inputs(#[comptime] config: Self::Config) -> (Self::LhsTile, Self::RhsTile);

    /// Reads the result of the accumulator and hands it to the stage writer
    ///
    /// # Quantization
    ///
    /// If some `quantization` is provided, the read will also requantize the stage in the output
    /// and update the scaling of the output tensor. This assumes that [execute] is called
    /// with some `scaling` provided.
    fn read_accumulator<Out: StageWriter<MP::EO>, G: global::GlobalConfig>(
        acc: &Self::Accumulator,
        out: &mut Out,
        quantization: CubeOption<IndexedQuantization<MP>>,
        #[comptime] stage_config: Self::Config,
        #[comptime] global_config: G,
    );

    /// Create an instance of the accumulator, without data
    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator;

    /// Fill the accumulator with zeros
    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] config: Self::Config);

    /// Fill the accumulator with data
    fn fill_accumulator<L: AccumulatorLoader<MP, Self::Config>>(
        loader: &mut L,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    );
}

#[cube]
/// Input to the stage matmul, responsible of handing slices of data
/// at precise locations in the stage
pub trait StageReader<ES: Numeric>: CubeType {
    /// Hands a portion of data from the stage, whose location is function of the
    /// plane, buffer and accumulator indexes.
    fn read_tile<S: StageConfig>(
        this: &Self,
        compute_plane_offset: u32,
        buffer_offset: u32,
        accumulator_offset: u32,
        #[comptime] config: S,
    ) -> Slice<Line<ES>>;
}

#[cube]
/// Responsible of writing the accumulated stage matmul output
/// to global memory
pub trait StageWriter<EO: Numeric>: CubeType + 'static + Send + Sync {
    /// Writes the given slice to global memory, at a position that depends on
    /// plane and accumulator indexes.
    fn write<ES: Numeric, G: global::GlobalConfig>(
        this: &mut Self,
        slice: Slice<Line<ES>>,
        compute_plane_offset: u32,
        accumulator_offset: u32,
        #[comptime] config: G,
    );
}

/// Configuration for the Stage matmul (SMM) level
pub trait StageConfig: MatmulConfig {
    /// Underlying Tile matmul config
    type TmmConfig: TileConfig;

    /// Convert itself to the underlying tile matmul config
    fn to_tmm_config(self) -> Self::TmmConfig;

    /// Returns the line size for the given ident
    fn line_size(&self, ident: Ident) -> u32;

    /// Returns the [StageTiling] for the given ident
    fn tiling_dimensions(&self, ident: Ident) -> TilingDimensions;

    /// Returns the [MatrixLayout] for the given ident
    fn matrix_layout(&self, ident: Ident) -> MatrixLayout;

    /// Returns the number of planes in the cube
    fn num_planes(&self) -> u32;

    /// Returns the size of the plane dimension
    fn plane_dim(&self) -> u32;

    fn tile_count(&self) -> &MatmulSize;

    fn buffering(&self) -> Buffering;
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Buffering {
    Single,
    Double,
}
