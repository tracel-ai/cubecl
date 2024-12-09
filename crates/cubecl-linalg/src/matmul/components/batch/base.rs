use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::batch;
use crate::matmul::components::global::args::{self, GmmArgs, TensorInput, TensorOutput};
use crate::matmul::components::{
    config::MatmulConfig, global, Ident, MatmulKernel, MatmulLaunch, StageDim,
};

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
pub trait Matmul<GA: GmmArgs<EG>, EG: Numeric>:
    'static + Send + Sync + MatmulKernel<EG, EG, Config: Config> + MatmulLaunch<EG, EG, GA>
{
    /// Performs batchwise matrix multiplication over tensors.
    fn execute(
        lhs: TensorInput<EG, GA>,
        rhs: TensorInput<EG, GA>,
        out: TensorOutput<EG, GA>,
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

#[cube(launch_unchecked)]
pub(crate) fn batch_matmul<GA: GmmArgs<EG>, EG: Numeric, BMM: batch::Matmul<GA, EG>>(
    inputs: &GA::Input,
    output: &mut GA::Output,
    #[comptime] config: BMM::Config,
) {
    let mut state = GA::init_state(inputs, output);

    let lhs = TensorInput::<EG, GA>::new(&state, args::Ident::Lhs);
    let rhs = TensorInput::<EG, GA>::new(&state, args::Ident::Rhs);
    let out = TensorOutput::<EG, GA>::new(&mut state);

    BMM::execute(lhs, rhs, out, config);
}
