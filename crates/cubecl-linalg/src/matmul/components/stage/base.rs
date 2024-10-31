use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::config::MatmulConfig;
use crate::matmul::components::matrix::{Ident, MatrixLayout};
use crate::matmul::components::stage_dim::StageDim;
use crate::matmul::components::{global, tile, MatmulKernel};

use super::tiling_order::TilingOrderConfig;

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
pub trait Matmul<I: Numeric, O: Numeric, Lhs: StageReader<I, S>, Rhs: StageReader<I, S>, S: Config>:
    'static + Send + Sync + MatmulKernel<I, O, Config = S>
{
    /// Number of rows of LHS
    const M: u32;
    /// Number of columns of RHS
    const N: u32;
    /// Common dimension of LHS and RHS
    const K: u32;

    /// Contains the matrix multiplication output, that can be shared across the different planes of the cube.
    /// The same Accumulator will be added to across multiple executions of the stage matmul.
    type Accumulator: CubeType;

    /// Executes the matrix multiplication of LHS and RHS, adding the result to the accumulator
    fn execute(lhs: &Lhs, rhs: &Rhs, acc: &mut Self::Accumulator, #[comptime] config: S);

    /// Creates an accumulator initialized to zeros
    fn acc_init_zeros(#[comptime] config: S) -> Self::Accumulator;

    /// Reads the result of the accumulator and hands it to the stage writer
    fn acc_read<Out: StageWriter<O, G>, G: global::Config>(
        acc: &Self::Accumulator,
        out: &mut Out,
        #[comptime] stage_config: S,
        #[comptime] global_config: G,
    );
}

#[cube]
/// Input to the stage matmul, responsible of handing slices of data
/// at precise locations in the stage
pub trait StageReader<ES: Numeric, S: Config>: CubeType {
    /// Hands a portion of data from the stage, whose location is function of the
    /// plane, buffer and accumulator indexes.
    fn read_tile(
        this: &Self,
        compute_plane_offset: u32,
        buffer_offset: u32,
        accumulator_offset: u32,
        #[comptime] config: S,
    ) -> &Slice<'_, Line<ES>>;
}

#[cube]
/// Responsible of writing the accumulated stage matmul output
/// to global memory
pub trait StageWriter<EG: Numeric, G: global::Config>: CubeType + 'static + Send + Sync {
    /// Writes the given slice to global memory, at a position that depends on
    /// plane and accumulator indexes.
    fn write<ES: Numeric>(
        this: &mut Self,
        slice: &Slice<'_, Line<ES>>,
        compute_plane_offset: u32,
        accumulator_offset: u32,
        #[comptime] config: G,
    );
}

/// Configuration for the Stage matmul (SMM) level
pub trait Config: MatmulConfig {
    /// Underlying Tile matmul config
    type TmmConfig: tile::Config;

    /// Convert itself to the underlying tile matmul config
    fn to_tmm_config(self) -> Self::TmmConfig;

    /// Returns the line size for the given ident
    fn line_size(&self, ident: Ident) -> u32;

    /// Returns the [StageDim] for the given ident
    fn stage_dim(&self, ident: Ident) -> StageDim;

    /// Returns the [MatrixLayout] for the given ident
    fn layout(&self, ident: Ident) -> MatrixLayout;

    /// Returns the number of planes in the cube
    fn num_planes(&self) -> u32;

    /// Returns the size of the plane dimension
    fn plane_dim(&self) -> u32;

    /// Returns the order in which tiles should be loaded to the stage
    fn tiling_order(&self) -> TilingOrderConfig;
}

pub trait StageSize: 'static + Send + Sync {
    const NUM_M: u32;
    const NUM_N: u32;
    const NUM_K: u32;
}

macro_rules! create_cmma_stage {
    ($name:ident, $m:expr, $n:expr, $k:expr) => {
        pub struct $name;

        impl StageSize for $name {
            const NUM_M: u32 = $m;
            const NUM_N: u32 = $n;
            const NUM_K: u32 = $k;
        }
    };
}

// This list is not exhaustive. Add what you need.
create_cmma_stage!(S1x1x1, 1, 1, 1);
create_cmma_stage!(S1x1x2, 1, 1, 2);
create_cmma_stage!(S1x2x1, 1, 2, 1);
create_cmma_stage!(S2x1x1, 2, 1, 1);
create_cmma_stage!(S2x2x1, 2, 2, 1);
create_cmma_stage!(S2x2x2, 2, 2, 2);
create_cmma_stage!(S4x4x1, 4, 4, 1);
create_cmma_stage!(S4x4x2, 4, 4, 2);
create_cmma_stage!(S8x1x1, 8, 1, 1);
create_cmma_stage!(S8x8x1, 8, 8, 1);
