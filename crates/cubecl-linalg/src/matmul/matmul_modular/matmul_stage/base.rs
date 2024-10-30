use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::matmul_modular::{matmul_global::GmmConfig, Matmul};

use super::{SmmConfig, StageReader, StageWriter};

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
pub trait StageMatmul<
    I: Numeric,
    O: Numeric,
    Lhs: StageReader<I, S>,
    Rhs: StageReader<I, S>,
    S: SmmConfig,
>: 'static + Send + Sync + Matmul<I, O, Config = S>
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
    fn acc_read<Out: StageWriter<O, G>, G: GmmConfig>(
        acc: &Self::Accumulator,
        out: &mut Out,
        #[comptime] stage_config: S,
        #[comptime] global_config: G,
    );
}
