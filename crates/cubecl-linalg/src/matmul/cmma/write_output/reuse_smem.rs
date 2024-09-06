use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{
    super::{
        base::{Dimensions, Ids, Offsets},
        config::CmmaComptimeInfo,
    },
    base::{shared_memory_to_output, OutputWriter},
};

pub(crate) struct ReuseSmemWriter;

#[cube]
impl OutputWriter for ReuseSmemWriter {
    fn write_to_output<F: Float>(
        out: &mut Tensor<F>,
        accumulators: Sequence<cmma::Matrix<F>>,
        offsets: Offsets,
        dims: Dimensions,
        config: Comptime<CmmaComptimeInfo>,
        ids: Ids,
    ) {
        let num_accumulators = Comptime::map(config, |c| c.num_accumulators);
        let tile_size = Comptime::map(config, |c| c.tile_size);
        let lane_dim = Comptime::map(config, |c| c.lane_dim);

        let sm_stride = tile_size * tile_size;
        let sm_size = lane_dim * sm_stride;

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

            shared_memory_to_output(out, offsets, ids.coop, acc_sm, dims, config, n, ids);
        }
    }
}
