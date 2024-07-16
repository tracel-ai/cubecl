use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{
    base::{Dimensions, Offsets, SharedMemories},
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
    shared_memories: SharedMemories<F, FC>,
    config: Comptime<CmmaConfig>,
    dims: Dimensions,
) {
    let block_size_k = Comptime::map(config, |c| c.block_size_k);
    let n_loops = calculate_n_loops::<F>(dims.k, config);

    for k in range(0u32, n_loops, Comptime::new(false)) {
        offsets.k = k * Comptime::runtime(block_size_k);

        load_to_shared_memories::<F, FC>(lhs, rhs, offsets, shared_memories, config, dims);

        sync_units();

        compute_loop::<F, FC>(shared_memories, config);

        sync_units();
    }

    write_to_output::<F>(out, shared_memories.accumulate, offsets, dims, config);
}

#[cube]
#[allow(unused_assignments)]
fn calculate_n_loops<F: Float>(dim_k: UInt, config: Comptime<CmmaConfig>) -> UInt {
    let block_size_k = Comptime::map(config, |c| c.block_size_k);
    let check_k_bounds = Comptime::map(config, |c| c.check_k_bounds);

    let mut n_loops = UInt::new(0); // TODO support syntax let x = if ... else ...
    if Comptime::get(check_k_bounds) {
        n_loops = UInt::cast_from(F::ceil(
            F::cast_from(dim_k) / F::cast_from(Comptime::runtime(block_size_k)),
        ));
    } else {
        n_loops = dim_k / Comptime::runtime(block_size_k);
    }

    n_loops
}
