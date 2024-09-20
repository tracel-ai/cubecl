use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{
    compute_loop::base::compute_loop,
    config::ComptimeCmmaInfo,
    load_shared_memory::base::load_to_shared_memories,
    runtime_info::{Fragments, RuntimeCmmaInfo, SharedMemories},
    write_output::base::write_to_output,
};

#[cube]
pub(crate) fn matmul_execute<F: Float, FC: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    out: &mut Tensor<F>,
    shared_memories: SharedMemories<FC>,
    mut fragments: Fragments<F, FC>,
    runtime_info: RuntimeCmmaInfo,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) {
    let b_k = comptime_info.block_size_k;
    let num_loops = (runtime_info.dims.k + b_k - 1) / b_k;

    for block in 0..num_loops {
        let k_offset = block * b_k;

        load_to_shared_memories::<F, FC>(
            lhs,
            rhs,
            shared_memories,
            k_offset,
            runtime_info,
            comptime_info,
        );

        sync_units();

        compute_loop::<F, FC>(
            shared_memories,
            &mut fragments,
            runtime_info.compute_ids,
            comptime_info,
        );

        sync_units();
    }

    write_to_output(out, fragments.accumulators, runtime_info, comptime_info);
}
