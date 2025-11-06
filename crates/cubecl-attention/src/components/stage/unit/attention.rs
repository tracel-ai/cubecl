use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::layout::Coords1d;

use crate::components::{
    global::simple::UnitAttentionWriter,
    stage::{
        UnitReducer, kv_reuse_attention::KVReuseStageAttention, partitioner::AttentionPartitioner,
        unit::UnitKVReuseStageConfig,
    },
    tile::FragmentAttention,
};

pub type UnitKVReuseStageAttention<AP, SK, SV, SO, FA> = KVReuseStageAttention<
    AP,
    SK,
    SV,
    SO,
    FA,
    UnitPartitioner,
    UnitKVReuseStageConfig<<FA as FragmentAttention<AP>>::Config>,
>;

pub struct UnitPartitioner {}

#[cube]
impl AttentionPartitioner for UnitPartitioner {
    type Reducer = UnitReducer;
    type Writer<ES: Float, EG: Float> = UnitAttentionWriter<ES, EG>;

    fn seq_q_index() -> Coords1d {
        UNIT_POS
    }
}
