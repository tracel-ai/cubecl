use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};

use crate::matmul::components::{
    Ident, InputIdent, MatmulConfigFactory, MatmulPrecision, MatmulSize, MatrixLayout,
    TilingDimensions,
    config::MatmulConfig,
    global::{self, AccumulatorLoader, GlobalWriter},
    tile::TileConfig,
};

use super::{StageEventListener, StageToTileReader, TilingLayout};

pub trait ReaderFamily: Send + Sync + 'static {
    type Reader<ES: Numeric, T: TilingLayout>: StageToTileReader<ES>;
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

    /// Returns the number of tiles in each axis of the stage.
    fn tile_shape(config: &Self::Config) -> MatmulSize;

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

    type Writer: GlobalWriter<MP::EO>;

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
        #[comptime] config: Self::Config,
        listener: SEL,
    );

    fn init_tile_inputs(#[comptime] config: Self::Config) -> (Self::LhsTile, Self::RhsTile);

    /// Create an instance of the accumulator, without data
    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator;

    /// Fill the accumulator with zeros
    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] config: Self::Config);

    /// Fill the accumulator with data
    fn fill_accumulator<L: AccumulatorLoader<MP>>(
        loader: &mut L,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    );

    fn init_writer(
        tensor: VirtualTensor<MP::EO, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
    ) -> Self::Writer;

    /// Reads the result of the accumulator and hands it to the stage writer
    ///
    /// # Quantization
    ///
    /// If some `quantization` is provided, the read will also requantize the stage in the output
    /// and update the scaling of the output tensor. This assumes that [execute] is called
    /// with some `scaling` provided.
    fn write_results<G: global::GlobalConfig>(
        acc: &Self::Accumulator,
        out: &mut Self::Writer,
        #[comptime] stage_config: Self::Config,
        #[comptime] global_config: G,
    );
}

/// Configuration for the Stage matmul (SMM) level
pub trait StageConfig: MatmulConfig {
    /// Underlying Tile matmul config
    type TmmConfig: TileConfig;

    /// Convert itself to the underlying tile matmul config
    fn to_tmm_config(self) -> Self::TmmConfig;

    /// Returns the line size for the given ident
    fn stage_line_size(&self, ident: Ident) -> u32;

    /// Returns the [StageTiling] for the given ident
    fn tiling_dimensions(&self, ident: Ident) -> TilingDimensions;

    /// Returns the [MatrixLayout] for the given ident
    fn matrix_layout(&self, ident: Ident) -> MatrixLayout;

    /// Returns the number of planes in the cube
    fn num_planes(&self) -> u32;

    /// Returns the size of the plane dimension
    fn plane_dim(&self) -> u32;

    fn tile_count(&self) -> &MatmulSize;

    fn buffering(&self) -> StageBuffering;

    fn num_stages(&self, ident: InputIdent) -> u32;

    /// Execution primitive = unit or plane, depending on the algorithm
    /// Returns a pair stating the dimension in m and n respectfully
    fn num_tmm_per_execution_primitive(&self) -> (u32, u32);
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum StageBuffering {
    Single,
    Double,
}
