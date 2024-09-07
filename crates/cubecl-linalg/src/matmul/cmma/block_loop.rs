use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{
    base::{Accumulators, Dimensions, Offsets, SharedMemories},
    compute_loop::compute_loop,
    config::CmmaConfig,
    load_shared_memory::load_to_shared_memories,
    write_output::write_to_output,
};

// #[cube]
// pub(crate) fn block_loop<F: Float, FC: Float>(
//     lhs: &Tensor<F>,
//     rhs: &Tensor<F>,
//     out: &mut Tensor<F>,
//     mut offsets: Offsets,
//     shared_memories: SharedMemories<FC>,
//     accumulators: Accumulators<F>,
//     #[comptime] config: CmmaConfig,
//     dims: Dimensions,
// ) {
//     let block_size_k = config.block_size_k;
//     let n_loops = (dims.k + block_size_k - 1) / block_size_k;

//     for block in 0u32..n_loops {
//         offsets.k = block * block_size_k;

//         load_to_shared_memories::<F, FC>(lhs, rhs, offsets, shared_memories, dims, config);

//         sync_units();

//         compute_loop::<F, FC>(shared_memories, accumulators, config);

//         sync_units();
//     }

//     write_to_output::<F>(out, accumulators, offsets, dims, config);
// }

// Recursive expansion of cube macro
// ==================================

#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn block_loop<F: Float, FC: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    out: &mut Tensor<F>,
    mut offsets: Offsets,
    shared_memories: SharedMemories<FC>,
    accumulators: Accumulators<F>,
    config: CmmaConfig,
    dims: Dimensions,
) {
    let block_size_k = config.block_size_k;
    let n_loops = (dims.k + block_size_k - 1) / block_size_k;
    for block in 0..n_loops {
        offsets.k = block * block_size_k;
        load_to_shared_memories::<F, FC>(lhs, rhs, offsets, shared_memories, dims, config);
        sync_units();
        compute_loop::<F, FC>(shared_memories, accumulators, config);
        sync_units();
    }
    write_to_output::<F>(out, accumulators, offsets, dims, config);
}
#[allow(clippy::module_inception)]
pub(crate) mod block_loop {
    use super::*;
    #[allow(unused, clippy::all)]
    pub fn expand<F: Float, FC: Float>(
        context: &mut cubecl::prelude::CubeContext,
        lhs: <Tensor<F> as cubecl::prelude::CubeType>::ExpandType,
        rhs: <Tensor<F> as cubecl::prelude::CubeType>::ExpandType,
        out: <Tensor<F> as cubecl::prelude::CubeType>::ExpandType,
        offsets: <Offsets as cubecl::prelude::CubeType>::ExpandType,
        shared_memories: <SharedMemories<FC> as cubecl::prelude::CubeType>::ExpandType,
        accumulators: <Accumulators<F> as cubecl::prelude::CubeType>::ExpandType,
        config: CmmaConfig,
        dims: <Dimensions as cubecl::prelude::CubeType>::ExpandType,
    ) -> <() as cubecl::prelude::CubeType>::ExpandType {
        {
            let block_size_k = config.block_size_k;
            let n_loops = {
                let _lhs = {
                    let _lhs = {
                        let _lhs = dims.clone().k.clone();
                        let _rhs = cubecl::frontend::ExpandElementTyped::from_lit(block_size_k);
                        cubecl::frontend::add::expand(context, _lhs, _rhs)
                    };
                    let _rhs = cubecl::frontend::ExpandElementTyped::from_lit(1);
                    cubecl::frontend::sub::expand(context, _lhs, _rhs)
                };
                let _rhs = cubecl::frontend::ExpandElementTyped::from_lit(block_size_k);
                cubecl::frontend::div::expand(context, _lhs, _rhs)
            };
            {
                let _start = cubecl::frontend::ExpandElementTyped::<u32>::from_lit(0);
                let _end = n_loops;
                let _range = cubecl::frontend::Range {
                    start: _start,
                    end: _end,
                    inclusive: false,
                };
                let _unroll = false;
                cubecl::frontend::branch::for_expand(context, _range, _unroll, |context, block| {
                    let _var = offsets.clone().k.clone();
                    let _value = {
                        let _lhs = block.clone();
                        let _rhs = cubecl::frontend::ExpandElementTyped::from_lit(block_size_k);
                        cubecl::frontend::mul::expand(context, _lhs, _rhs)
                    };
                    cubecl::frontend::assign::expand(context, _value, _var);
                    {
                        let _arg_0 = lhs.clone();
                        let _arg_1 = rhs.clone();
                        let _arg_2 = offsets.clone();
                        let _arg_3 = shared_memories.clone();
                        let _arg_4 = dims.clone();
                        let _arg_5 = config;
                        load_to_shared_memories::expand::<F, FC>(
                            context, _arg_0, _arg_1, _arg_2, _arg_3, _arg_4, _arg_5,
                        )
                    };
                    {
                        sync_units::expand(context)
                    };
                    {
                        let _arg_0 = shared_memories.clone();
                        let _arg_1 = accumulators.clone();
                        let _arg_2 = config;
                        compute_loop::expand::<F, FC>(context, _arg_0, _arg_1, _arg_2)
                    };
                    {
                        sync_units::expand(context)
                    };
                    ()
                });
            };
            {
                let _arg_0 = out.clone();
                let _arg_1 = accumulators.clone();
                let _arg_2 = offsets.clone();
                let _arg_3 = dims.clone();
                let _arg_4 = config;
                write_to_output::expand::<F>(context, _arg_0, _arg_1, _arg_2, _arg_3, _arg_4)
            };
            ()
        }
    }
}
