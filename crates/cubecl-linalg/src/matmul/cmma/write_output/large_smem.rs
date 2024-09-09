use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma::base::RuntimeCmmaInfo;

use super::{
    super::config::ComptimeCmmaInfo,
    base::{shared_memory_to_output, OutputWriter},
};

pub(crate) struct LargeSmemWriter;

#[cube]
impl OutputWriter for LargeSmemWriter {
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

        let smem_stride = tile_size * tile_size;
        let smem_stride_r = Comptime::runtime(smem_stride);
        let smem_size = num_accumulators * num_coops * smem_stride;

        let mut acc_sm = SharedMemory::<F>::new(Comptime::get(smem_size));

        let slice_offset = ids.coop * Comptime::runtime(num_accumulators * smem_stride);
        let smem_position_base = Comptime::runtime(num_accumulators) * ids.coop;

        for n in range(0u32, Comptime::get(num_accumulators), Comptime::new(true)) {
            let slice_start = slice_offset + n * smem_stride_r;
            let slice_end = slice_start + smem_stride_r;

            let slice = acc_sm.slice_mut(slice_start, slice_end);

            cmma::store::<F>(
                slice,
                accumulators.index(n),
                UInt::new(16),
                cmma::MatrixLayout::RowMajor,
            );
        }

        for n in range(0u32, Comptime::get(num_accumulators), Comptime::new(true)) {
            let smem_position = smem_position_base + n;
            shared_memory_to_output(out, smem_position, acc_sm, n, runtime_info, comptime_info);
        }
    }
}
