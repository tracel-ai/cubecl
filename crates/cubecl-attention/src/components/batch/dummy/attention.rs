use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};
use std::marker::PhantomData;

use crate::components::{
    AttentionPrecision,
    batch::{
        BatchAttention, BatchAttentionConfig, CubeCountInput, dummy::config::DummyBatchConfig,
    },
    global::GlobalAttention,
};

pub struct DummyBatchAttention<AP: AttentionPrecision, GA: GlobalAttention<AP>> {
    _phantom: PhantomData<(AP, GA)>,
}

#[cube]
impl<GA: GlobalAttention<AP>, AP: AttentionPrecision> BatchAttention<AP>
    for DummyBatchAttention<AP, GA>
{
    type Config = DummyBatchConfig<GA::Config>;

    fn execute(
        query: VirtualTensor<AP::EI>,
        key: VirtualTensor<AP::EI>,
        value: VirtualTensor<AP::EI>,
        out: VirtualTensor<AP::EO, ReadWrite>,
        cube_count_args: CubeCountInput,
        #[comptime] config: Self::Config,
    ) {
        // let n = config.seq_k();
        // There are ceil(n/br) to launch for each head and batch
        // Compute offsets
        // But for now we assume every parameter = 8 so it's only one time the calls to tile matmul

        GA::execute(
            GA::init_query_loader(query),
            GA::init_key_loader(key, config.global_config()),
            GA::init_value_loader(value, config.global_config()),
            GA::init_writer(),
            &mut GA::init_accumulator(),
            config.global_config(),
        )
    }
}
