use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::global::args::{self, MatmulArgs, TensorInput, TensorOutput};
use crate::matmul::components::{config::MatmulConfig, global, Ident, MatmulLaunch, StageDim};
use crate::matmul::components::{MatmulSpec, SingleMatmulSpec};

pub trait BatchMatmulFamily: 'static + Send + Sync + MatmulLaunch<Config: Config> {
    type Matmul<MS: MatmulSpec>: BatchMatmul<MS, Config = Self::Config>;
}

#[cube]
/// Provides matrix multiplication operations at the batch level.
///
/// At the batch level,
///  - Inputs are whole tensors in global memory.
///  - All Cubes can collaborate to solve the problem
///  - Dimensions M, N and K can be arbitrary large,
///    as well as the number of batches.
///
/// # Assumptions
/// - Line sizes of the inputs evenly divide the dimension they are aligned with.
/// - Enough Cubes are launched to perform the whole computation.
///
/// # Safety
///
/// It is not assumed that the matmul's dimensions match its inputs dimensions perfectly.
/// It is therefore important to use an underlying global matmul that performs check bounds,
/// and to not launch more Cubes than necessary.
pub trait BatchMatmul<MS: MatmulSpec>: 'static + Send + Sync {
    type Config: Config;

    /// Performs batchwise matrix multiplication over tensors.
    fn execute(
        lhs: TensorInput<MS::EG, MS::Args>,
        rhs: TensorInput<MS::EG, MS::Args>,
        out: TensorOutput<MS::EG, MS::Args>,
        #[comptime] config: Self::Config,
    );
}

/// Configuration for the Batch matmul (BMM) level
pub trait Config: MatmulConfig {
    /// Underlying Global matmul config
    type GmmConfig: global::Config;

    /// Convert itself to the underlying global matmul config
    fn to_gmm_config(&self) -> Self::GmmConfig;

    /// Returns the [StageDim] for the given ident
    fn stage_dim(&self, ident: Ident) -> Box<dyn StageDim>;

    /// Returns the largest m dimension supported with these configs
    fn max_m(&self) -> u32;

    /// Returns the largest n dimension supported with these configs
    fn max_n(&self) -> u32;

    /// Returns the largest number of batches supported with these configs
    fn max_batches(&self) -> u32;
}

type Input<Args, EG> = <Args as MatmulArgs>::Input<EG>;
type Output<Args, EG> = <Args as MatmulArgs>::Output<EG>;

type BMatmul<EG, ES, EA, Args, BMM> =
    <BMM as BatchMatmulFamily>::Matmul<SingleMatmulSpec<32, EG, ES, EA, Args>>;

#[cube(launch_unchecked)]
pub(crate) fn matmul<
    EG: Numeric,
    ES: Numeric,
    EA: Numeric,
    Args: MatmulArgs,
    BMM: BatchMatmulFamily,
>(
    inputs: &Input<Args, EG>,
    output: &mut Output<Args, EG>,
    #[comptime] config: BMM::Config,
) {
    let mut state = Args::init_state(inputs, output);

    let lhs = TensorInput::<EG, Args>::new(&state, args::TensorInputIdent::Lhs);
    let rhs = TensorInput::<EG, Args>::new(&state, args::TensorInputIdent::Rhs);
    let out = TensorOutput::<EG, Args>::new(&mut state);

    BMatmul::<EG, ES, EA, Args, BMM>::execute(lhs, rhs, out, config);
}
