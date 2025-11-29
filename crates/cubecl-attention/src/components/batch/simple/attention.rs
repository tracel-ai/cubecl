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
    stage::StageAttentionConfig as _,
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
        let global_config = config.global_config();
        let q_index = CUBE_POS_X;
        let batch_index = CUBE_POS_Y;

        let stage_q_offset = q_index * global_config.stage_config().elements_in_stage_seq_q();

        // Assume [batch, num_heads, seq_*, head_dim] layout
        let seq_q = query.shape(2);
        let seq_kv = key.shape(2);

        GA::execute(
            GA::init_query_reader(batch_index, stage_q_offset, query, global_config),
            GA::init_key_reader(batch_index, key, global_config),
            GA::init_value_reader(batch_index, value, global_config),
            GA::init_mask_reader(batch_index, stage_q_offset, mask, seq_kv, global_config),
            GA::init_writer(batch_index, stage_q_offset, out, global_config),
            seq_q,
            seq_kv,
            config.global_config(),
        )
    }
}
