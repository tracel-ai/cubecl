use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::super::prologue::RuntimeCmmaInfo;

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
        #[comptime] comptime_info: ComptimeCmmaInfo,
    ) {
        let num_accumulators = comptime_info.num_accumulators;
        let num_compute_planes = comptime_info.num_compute_planes;
        let plane_id = runtime_info.compute_ids.plane;

        let smem_stride = comptime_info.tile_size_m * comptime_info.tile_size_n;
        let smem_size = num_accumulators * num_compute_planes * smem_stride;

        let mut acc_sm = SharedMemory::<F>::new(smem_size);

        let slice_offset = plane_id * num_accumulators * smem_stride;
        let smem_position_base = num_accumulators * plane_id;

        #[unroll]
        for n in 0..num_accumulators {
            let slice_start = slice_offset + n * smem_stride;
            let slice_end = slice_start + smem_stride;

            let slice = acc_sm.slice_mut(slice_start, slice_end);

            cmma::store::<F>(
                slice,
                accumulators.index(n),
                comptime_info.tile_size_n,
                cmma::MatrixLayout::RowMajor,
            );
        }

        #[unroll]
        for n in 0..num_accumulators {
            let smem_position = smem_position_base + n;
            shared_memory_to_output(out, smem_position, acc_sm, n, runtime_info, comptime_info);
        }
    }
}
