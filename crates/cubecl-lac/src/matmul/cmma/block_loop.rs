use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{
    base::{Accumulators, Dimensions, Offsets, SharedMemories},
    compute_loop::compute_loop,
    config::CmmaConfig,
    load_shared_memory::load_to_shared_memories,
    write_output::write_to_output,
};

#[cube]
pub(crate) fn block_loop<F: Float, FC: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    out: &mut Tensor<F>,
    mut offsets: Offsets,
    shared_memories: SharedMemories<FC>,
    accumulators: Accumulators<F>,
    config: Comptime<CmmaConfig>,
    dims: Dimensions,
) {
    let block_size_k = Comptime::map(config, |c| c.block_size_k);
    let n_loops = dims.k / Comptime::runtime(block_size_k); // TODO not always true if check_k_bounds

    for k in range(0u32, n_loops, Comptime::new(false)) {
        offsets.k = k * Comptime::runtime(block_size_k);

        load_to_shared_memories::<F, FC>(lhs, rhs, offsets, shared_memories, config, dims);

        sync_units();

        compute_loop::<F, FC>(shared_memories, accumulators, config);

        sync_units();
    }

    write_to_output::<F>(out, accumulators, offsets, dims, config);
}
