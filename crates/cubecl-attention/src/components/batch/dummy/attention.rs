use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::VirtualTensor;
use std::marker::PhantomData;

use crate::components::{
    AttentionPrecision,
    batch::{
        BatchAttention, BatchAttentionConfig, CubeCountInput, dummy::config::DummyBatchConfig,
    },
    global::{GlobalAttention, GlobalAttentionConfig as _},
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
        _cube_count_args: CubeCountInput,
        #[comptime] config: Self::Config,
    ) {
        let q_index = CUBE_POS;

        let q_offset = q_index * config.global_config().tiling_scheme().seq_q();
        let seq_kv = key.shape(1);

        let global_config = config.global_config();
        GA::execute(
            GA::init_query_loader(q_offset, query, global_config),
            GA::init_key_loader(key, global_config),
            GA::init_value_loader(value, global_config),
            GA::init_writer(q_offset, out, global_config),
            seq_kv,
            config.global_config(),
        )
    }
}
