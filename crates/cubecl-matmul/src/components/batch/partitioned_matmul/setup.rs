use std::marker::PhantomData;

use crate::components::batch::entry_point::matmul;
use crate::components::batch::partitioned_matmul::config::PartitionedBatchConfig;
use crate::components::batch::partitioned_matmul::matmul::PartitionedBatchMatmul;
use crate::components::batch::partitioned_matmul::partition::GlobalPartitionMatmul;
use crate::components::batch::{BatchMatmulFamily, CubeCountInputArgs};
use crate::components::global::GlobalMatmulFamily;
use crate::components::{
    Args, EA, EO, InputRuntimeArg, LhsG, LhsS, MatmulPrecision, MatmulProblem, MatmulSelection,
    MatmulSpec, OutputRuntimeArg, RhsG, RhsS,
};
use crate::components::{MatmulLineSizes, MatmulSetupError};
use cubecl_core::prelude::*;

/// Simple partitioned batch matmul family for any precision
pub struct PartitionedBatchMatmulFamily<GMM: GlobalMatmulFamily, S: GlobalPartitionMatmul> {
    _gmm: PhantomData<GMM>,
    _s: PhantomData<S>,
}

impl<GMM: GlobalMatmulFamily, S: GlobalPartitionMatmul> BatchMatmulFamily
    for PartitionedBatchMatmulFamily<GMM, S>
{
    type Matmul<MP: MatmulPrecision> = PartitionedBatchMatmul<MP, GMM::Matmul<MP>, S>;
    type Config = PartitionedBatchConfig<GMM::Config>;

    fn setup<MP: MatmulPrecision, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        line_sizes: &MatmulLineSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        let global_config = GMM::setup::<MP, R>(client, problem, selection, line_sizes)?;

        PartitionedBatchConfig::new(
            global_config,
            selection
                .hypercube_selection
                .to_hypercube_config(problem, client.properties().hardware.max_cube_count.clone()),
        )
        .validate(problem)
    }

    unsafe fn launch_unchecked<'a, MS: MatmulSpec, R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        input: InputRuntimeArg<'a, MS, R>,
        output: OutputRuntimeArg<'a, MS, R>,
        cube_count_input: CubeCountInputArgs<'a, R>,
        config: Self::Config,
    ) {
        unsafe {
            matmul::launch_unchecked::<
                Args<MS>,
                LhsG<MS>,
                RhsG<MS>,
                LhsS<MS>,
                RhsS<MS>,
                EA<MS>,
                EO<MS>,
                Self,
                R,
            >(
                client,
                cube_count,
                cube_dim,
                input,
                output,
                cube_count_input,
                config,
            );
        }
    }
}
