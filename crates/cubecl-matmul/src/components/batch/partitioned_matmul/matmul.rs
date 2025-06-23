use std::marker::PhantomData;

use crate::components::MatmulPrecision;
use crate::components::batch::partitioned_matmul::config::PartitionedBatchConfig;
use crate::components::batch::partitioned_matmul::partition::{
    GlobalPartitionMatmul, PartitionRangeDim, PartitionRanges,
};
use crate::components::batch::{BatchConfig as _, BatchMatmul, CubeCountArgs, CubeCounter};
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
        cube_count_args: CubeCountArgs,
        #[comptime] config: Self::Config,
    ) {
        // TODO maybe reads too many things
        let lhs_rank = lhs.rank();
        let rhs_rank = rhs.rank();
        let out_rank = out.rank();

        let problem_m = lhs.shape(lhs_rank - 2);
        let problem_n = rhs.shape(rhs_rank - 1);

        let mut problem_batch = 0;
        for o in 0..out_rank - 2 {
            problem_batch *= out.shape(o);
        }

        let cube_counter = CubeCounter::new(
            problem_m,
            problem_n,
            problem_batch,
            cube_count_args,
            config.cube_counter_config(),
        );

        let problem_k = lhs.shape(lhs_rank - 1);
        let k_range = (0, problem_k);

        let tiling_scheme = config.tiling_scheme();
        let (m_index, n_index, batch_index) = cube_counter.cube_pos_to_tensor_pos();

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
