use std::marker::PhantomData;

use crate::components::AvailableLineSizes;
use crate::components::InvalidConfigError;
use crate::components::MatmulChecker;
use crate::components::batch::BatchConfig as _;
use crate::components::batch::BatchMatmulFamily;
use crate::components::batch::entry_point::matmul;
use crate::components::batch::matmul::config::PartitionedBatchConfig;
use crate::components::batch::matmul::matmul::PartitionedBatchMatmul;
use crate::components::batch::matmul::partition::GlobalPartitionMatmul;
use crate::components::batch::matmul::partitioner::Partitioner;
use crate::components::global::GlobalMatmulFamily;
use crate::components::{
    Args, EA, EI, EO, ES, InputRuntimeArg, MatmulPrecision, MatmulProblem, MatmulSpec,
    OutputRuntimeArg,
};
use crate::kernels::MatmulAvailabilityError;
use crate::kernels::MatmulSetupError;
use crate::kernels::matmul::MatmulSelection;
use cubecl_core::prelude::*;

pub struct PartitionedBatchMatmulFamily<
    GMM: GlobalMatmulFamily,
    S: GlobalPartitionMatmul,
    P: Partitioner,
> {
    _gmm: PhantomData<GMM>,
    _s: PhantomData<S>,
    _c: PhantomData<P>,
}

impl<GMM: GlobalMatmulFamily, S: GlobalPartitionMatmul, P: Partitioner> BatchMatmulFamily
    for PartitionedBatchMatmulFamily<GMM, S, P>
{
    type Matmul<MP: MatmulPrecision> = PartitionedBatchMatmul<MP, GMM::Matmul<MP>, S, P>;
    type Input = GMM::Input;

    fn setup(
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        available_line_sizes: &mut AvailableLineSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        let global_config = GMM::setup(problem, selection, available_line_sizes)?;

        let elements_in_m = selection.tiling_scheme.elements_in_global_partition_m();
        let elements_in_n = selection.tiling_scheme.elements_in_global_partition_n();

        let cube_count = P::create_cube_count(
            (problem.m as u32).div_ceil(elements_in_m),
            (problem.n as u32).div_ceil(elements_in_n),
            (problem.num_batches() as u32)
                .div_ceil(selection.tiling_scheme.global_partition_size.batches),
        );

        Ok(PartitionedBatchConfig::new(global_config, cube_count))
    }

    unsafe fn launch_unchecked<'a, MS: MatmulSpec, R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        input: InputRuntimeArg<'a, MS, R>,
        output: OutputRuntimeArg<'a, MS, R>,
        config: Self::Config,
    ) {
        unsafe {
            matmul::launch_unchecked::<Args<MS>, EI<MS>, ES<MS>, EA<MS>, EO<MS>, Self, R>(
                client, cube_count, cube_dim, input, output, config,
            );
        }
    }
}

impl<GMM: GlobalMatmulFamily, S: GlobalPartitionMatmul, P: Partitioner> MatmulChecker
    for PartitionedBatchMatmulFamily<GMM, S, P>
{
    type Config = PartitionedBatchConfig<GMM::Config, P>;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        GMM::check_config(&config.global_config())
    }

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &Self::Config,
    ) -> Result<(), MatmulAvailabilityError> {
        GMM::check_availability::<R, MP>(client, &config.global_config())
    }
}
