use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma::base::RuntimeCmmaInfo;

use super::{
    super::config::ComptimeCmmaInfo,
    base::{shared_memory_to_output, OutputWriter},
};

pub(crate) struct ReuseSmemWriter;

#[cube]
impl OutputWriter for ReuseSmemWriter {
    fn write_to_output<F: Float>(
        out: &mut Tensor<F>,
        accumulators: Sequence<cmma::Matrix<F>>,
        runtime_info: RuntimeCmmaInfo,
        #[comptime] comptime_info: ComptimeCmmaInfo,
    ) {
        let num_accumulators = comptime_info.num_accumulators;
        let tile_size = comptime_info.tile_size;
        let num_coops = comptime_info.num_coops;
        let coop_id = runtime_info.compute_ids.coop;

        let sm_stride = tile_size * tile_size;
        let sm_size = num_coops * sm_stride;

        let acc_sm = SharedMemory::<F>::new(sm_size);

        let slice_offset = coop_id * sm_stride;
        let slice = acc_sm.slice_mut_unsafe(slice_offset, slice_offset + sm_stride);

        #[unroll]
        for n in 0..num_accumulators {
            cmma::store::<F>(
                slice,
                accumulators.index(n),
                16,
                cmma::MatrixLayout::RowMajor,
            );

            shared_memory_to_output(out, coop_id, acc_sm, n, runtime_info, comptime_info);
        }
    }
}
