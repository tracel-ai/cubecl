use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{
    base::{Dimensions, Ids, Offsets, SharedMemories},
    compute_loop::compute_loop,
    config::CmmaComptimeInfo,
    load_shared_memory::load_to_shared_memories,
    write_output::{base::OutputWriter, large_smem::LargeSmemWriter, reuse_smem::ReuseSmemWriter},
};

#[cube]
pub(crate) fn block_loop<F: Float, FC: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    out: &mut Tensor<F>,
    offsets: Offsets,
    shared_memories: SharedMemories<FC>,
    mut accumulators: Sequence<cmma::Matrix<F>>,
    config: Comptime<CmmaComptimeInfo>,
    dims: Dimensions,
    ids: Ids,
) {
    let block_size_k = Comptime::runtime(Comptime::map(config, |c| c.block_size_k));
    let write_out_reuse_smem = Comptime::map(config, |c| c.write_out_reuse_smem);

    // Equals ceil(dims.k / block_size_k)
    let num_loops = (dims.k + block_size_k - 1) / block_size_k;

    for block in range(0u32, num_loops, Comptime::new(false)) {
        let k_offset = block * block_size_k;

        load_to_shared_memories::<F, FC>(
            lhs,
            rhs,
            offsets,
            k_offset,
            shared_memories,
            dims,
            config,
            ids,
        );

        sync_units();

        compute_loop::<F, FC>(shared_memories, &mut accumulators, config, ids);

        sync_units();
    }

    if Comptime::get(write_out_reuse_smem) {
        ReuseSmemWriter::write_to_output(out, accumulators, offsets, dims, config, ids);
    } else {
        LargeSmemWriter::write_to_output(out, accumulators, offsets, dims, config, ids);
    }
}
