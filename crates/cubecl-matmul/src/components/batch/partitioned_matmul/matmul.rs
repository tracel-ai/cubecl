use std::marker::PhantomData;

use crate::components::batch::{BatchConfig as _, BatchMatmul, CubeCountInput};
use crate::components::global::{self, GlobalConfig, GlobalMatmul};
use crate::components::stage::StageConfig as _;
use crate::components::{AccG, batch::partitioned_matmul::config::PartitionedBatchConfig};
use crate::components::{LhsG, MatmulPrecision, RhsG};
use crate::components::{
    batch::partitioned_matmul::partition::{
        GlobalPartitionMatmul, PartitionRangeDim, PartitionRanges,
    },
    global::args::MatmulArgs,
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

/// Executes matrix multiplication at the batch level,
/// assigning each cube to handle multiple global matmuls.
///
/// Each cube performs a number of global matmuls specified by
/// the global partition size of the tiling scheme
pub struct PartitionedBatchMatmul<
    MP: MatmulPrecision,
    GMM: global::GlobalMatmul<MP>,
    S: GlobalPartitionMatmul,
> {
    _mp: PhantomData<MP>,
    _gmm: PhantomData<GMM>,
    _s: PhantomData<S>,
}

#[cube]
impl<MP: MatmulPrecision, GMM: GlobalMatmul<MP>, GPMM: GlobalPartitionMatmul> BatchMatmul<MP>
    for PartitionedBatchMatmul<MP, GMM, GPMM>
{
    type Config = PartitionedBatchConfig<GMM::Config>;

    fn execute<Args: MatmulArgs>(
        state: &mut Args::State<LhsG<MP>, RhsG<MP>, AccG<MP>>,
        cube_count_args: CubeCountInput,
        #[comptime] config: Self::Config,
    ) {
        let (_, _, problem_k) = Args::view_lhs(state).shape();
        let k_range = (0, problem_k);

        let (m_index, n_index, batch_index) =
            cube_count_args.cube_pos_to_tensor_pos(config.hypercube_config().global_order);

        let ranges = PartitionRanges::new(
            PartitionRangeDim::new(
                m_index,
                config.global_config().stage_config().elements_in_stage_m(),
                config.global_partition_size.m,
            ),
            PartitionRangeDim::new(
                n_index,
                config.global_config().stage_config().elements_in_stage_n(),
                config.global_partition_size.n,
            ),
            PartitionRangeDim::new(batch_index, 1u32, config.global_partition_size.batches),
        );

        let global_config = config.global_config();

        GPMM::execute::<Args, MP, GMM>(state, ranges, k_range, global_config);
    }
}
