use std::marker::PhantomData;

use crate::components::batch::partitioned_matmul::config::PartitionedBatchConfig;
use crate::components::batch::partitioned_matmul::matmul::PartitionedBatchMatmul;
use crate::components::batch::partitioned_matmul::partition::GlobalPartitionMatmul;
use crate::components::batch::{BatchMatmulFamily, CubeCountInputArgs, entry_point};
use crate::components::global::GlobalMatmulFamily;
use crate::components::global::args::MatmulArgs;
use crate::components::{
    InputRuntimeArg, MatmulElems, MatmulPrecision, MatmulProblem, MatmulSelection, OutputRuntimeArg,
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

    fn setup<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        line_sizes: &MatmulLineSizes,
        dtypes: &MatmulElems,
    ) -> Result<Self::Config, MatmulSetupError> {
        let global_config = GMM::setup(client, problem, selection, line_sizes, dtypes)?;

        PartitionedBatchConfig::new(
            global_config,
            selection
                .hypercube_selection
                .to_hypercube_config(problem, client.properties().hardware.max_cube_count.clone()),
            selection.tiling_scheme.global_partition_size,
        )
        .validate(problem)
    }

    unsafe fn launch_unchecked<'a, MA: MatmulArgs, R: Runtime>(
        client: &ComputeClient<R>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        input: InputRuntimeArg<'a, MA, R>,
        output: OutputRuntimeArg<'a, MA, R>,
        cube_count_input: CubeCountInputArgs<'a, R>,
        config: Self::Config,
        dtypes: &MatmulElems,
    ) -> Result<(), LaunchError> {
        unsafe {
            entry_point::matmul::launch_unchecked::<MA, Self, R>(
                client,
                cube_count,
                cube_dim,
                input,
                output,
                cube_count_input,
                config,
                [*dtypes.lhs_global, *dtypes.rhs_global, *dtypes.acc_global],
                [*dtypes.lhs_stage, *dtypes.rhs_stage, *dtypes.acc_stage],
                [
                    *dtypes.lhs_register,
                    *dtypes.rhs_register,
                    *dtypes.acc_register,
                ],
            )
        }
    }
}
