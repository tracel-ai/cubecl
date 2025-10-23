use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{CubeOption, tensor::r#virtual::VirtualTensor};
use std::marker::PhantomData;

use crate::components::{
    AttentionPrecision,
    attention_types::*,
    batch::{
        BatchAttention, BatchAttentionConfig, CubeCountInput, simple::config::SimpleBatchConfig,
    },
    global::{GlobalAttention, GlobalAttentionConfig as _},
};

pub struct SimpleBatchAttention<AP: AttentionPrecision, GA: GlobalAttention<AP>> {
    _phantom: PhantomData<(AP, GA)>,
}

#[cube]
impl<GA: GlobalAttention<AP>, AP: AttentionPrecision> BatchAttention<AP>
    for SimpleBatchAttention<AP, GA>
{
    type Config = SimpleBatchConfig<GA::Config>;

    fn execute(
        query: VirtualTensor<QG<AP>>,
        key: VirtualTensor<KG<AP>>,
        value: VirtualTensor<VG<AP>>,
        mask: CubeOption<VirtualTensor<MSK<AP>>>,
        out: VirtualTensor<OG<AP>, ReadWrite>,
        _cube_count_args: CubeCountInput,
        #[comptime] config: Self::Config,
    ) {
        let q_index = CUBE_POS;

        let q_offset = q_index
            * config
                .global_config()
                .tiling_scheme()
                .elements_in_stage_seq_q();

        let seq_q = query.shape(1);
        let seq_kv = key.shape(1);

        let global_config = config.global_config();
        GA::execute(
            GA::init_query_reader(q_offset, query, global_config),
            GA::init_key_reader(key, global_config),
            GA::init_value_reader(value, global_config),
            GA::init_mask_reader(q_offset, mask, seq_kv, global_config),
            GA::init_writer(q_offset, out, global_config),
            seq_q,
            seq_kv,
            config.global_config(),
        )
    }
}
