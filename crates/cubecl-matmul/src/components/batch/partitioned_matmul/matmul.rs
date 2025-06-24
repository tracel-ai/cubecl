use std::marker::PhantomData;

use crate::components::MatmulPrecision;
use crate::components::batch::partitioned_matmul::config::PartitionedBatchConfig;
use crate::components::batch::partitioned_matmul::partition::{
    GlobalPartitionMatmul, PartitionRangeDim, PartitionRanges,
};
use crate::components::batch::{BatchConfig as _, BatchMatmul, CubeDistribution};
use crate::components::global;
use crate::components::global::Quantization;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::CubeOption;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};

/// Executes matrix multiplication at the batch level,
/// assigning each cube to handle multiple global matmuls.
///
/// The algorithm supports any number of cubes,
/// looping as needed to process all data.
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
impl<MP: MatmulPrecision, GMM: global::GlobalMatmul<MP>, GPMM: GlobalPartitionMatmul>
    BatchMatmul<MP> for PartitionedBatchMatmul<MP, GMM, GPMM>
{
    type Config = PartitionedBatchConfig<GMM::Config>;

    fn execute(
        lhs: VirtualTensor<MP::EI>,
        rhs: VirtualTensor<MP::EI>,
        out: VirtualTensor<MP::EO, ReadWrite>,
        quantization: CubeOption<Quantization<MP>>,
        cube_count_args: CubeDistribution,
        #[comptime] config: Self::Config,
    ) {
        let lhs_rank = lhs.rank();

        let problem_k = lhs.shape(lhs_rank - 1);
        let k_range = (0, problem_k);

        let tiling_scheme = config.tiling_scheme();
        let (m_index, n_index, batch_index) =
            cube_count_args.cube_pos_to_tensor_pos(config.hypercube_config().global_order);

        let ranges = PartitionRanges::new(
            PartitionRangeDim::new(
                m_index,
                tiling_scheme.elements_in_stage_m(),
                tiling_scheme.elements_in_global_partition_m(),
            ),
            PartitionRangeDim::new(
                n_index,
                tiling_scheme.elements_in_stage_n(),
                tiling_scheme.elements_in_global_partition_n(),
            ),
            PartitionRangeDim::new(
                batch_index,
                1u32,
                tiling_scheme.global_partition_size.batches,
            ),
        );

        let global_config = config.global_config();
        let acc = GMM::init_accumulator(global_config);

        GPMM::execute::<MP, GMM>(
            lhs,
            rhs,
            out,
            ranges,
            acc,
            k_range,
            quantization,
            global_config,
        );
    }
}
