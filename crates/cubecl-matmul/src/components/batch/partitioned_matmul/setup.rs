use std::marker::PhantomData;

use crate::components::MatmulLineSizes;
use crate::components::batch::BatchMatmulFamily;
use crate::components::batch::entry_point::matmul;
use crate::components::batch::partitioned_matmul::config::PartitionedBatchConfig;
use crate::components::batch::partitioned_matmul::matmul::PartitionedBatchMatmul;
use crate::components::batch::partitioned_matmul::partition::GlobalPartitionMatmul;
use crate::components::batch::partitioned_matmul::partitioner::Partitioner;
use crate::components::global::GlobalMatmulFamily;
use crate::components::{
    Args, EA, EI, EO, ES, InputRuntimeArg, MatmulPrecision, MatmulProblem, MatmulSpec,
    OutputRuntimeArg,
};
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
    type Config = PartitionedBatchConfig<GMM::Config, Self::Partitioner>;
    type Partitioner = P;

    fn setup<MP: MatmulPrecision, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        line_sizes: MatmulLineSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        let global_config = GMM::setup::<MP, R>(client, problem, selection, line_sizes)?;

        PartitionedBatchConfig::new(global_config)
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
