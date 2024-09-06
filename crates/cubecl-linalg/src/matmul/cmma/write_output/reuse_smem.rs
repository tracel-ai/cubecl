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
        comptime_info: Comptime<ComptimeCmmaInfo>,
    ) {
        let num_accumulators = Comptime::map(comptime_info, |c| c.num_accumulators);
        let tile_size = Comptime::map(comptime_info, |c| c.tile_size);
        let num_coops = Comptime::map(comptime_info, |c| c.num_coops);
        let ids = runtime_info.ids;

        let sm_stride = tile_size * tile_size;
        let sm_size = num_coops * sm_stride;

        let acc_sm = SharedMemory::<F>::new(Comptime::get(sm_size));

        let slice_offset = ids.coop * Comptime::runtime(sm_stride);
        let slice =
            acc_sm.slice_mut_unsafe(slice_offset, slice_offset + Comptime::runtime(sm_stride));

        for n in range(0u32, Comptime::get(num_accumulators), Comptime::new(true)) {
            cmma::store::<F>(
                slice,
                accumulators.index(n),
                UInt::new(16),
                cmma::MatrixLayout::RowMajor,
            );

            shared_memory_to_output(out, ids.coop, acc_sm, n, runtime_info, comptime_info);
        }
    }
}
