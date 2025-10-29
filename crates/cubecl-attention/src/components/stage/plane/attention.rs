use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::layout::Coords1d;

use crate::components::{
    fragment::FragmentAttention,
    stage::{
        kv_reuse_attention::KVReuseStageAttention, partitioner::AttentionPartitioner,
        plane::PlaneKVReuseStageConfig,
    },
    tile::BroadcastReducer,
};

pub type PlaneKVReuseStageAttention<AP, SK, SV, SO, FA> = KVReuseStageAttention<
    AP,
    SK,
    SV,
    SO,
    FA,
    PlanePartitioner,
    PlaneKVReuseStageConfig<<FA as FragmentAttention<AP>>::Config>,
>;

pub struct PlanePartitioner {}

#[cube]
impl AttentionPartitioner for PlanePartitioner {
    type Reducer = BroadcastReducer;

    fn seq_q_index() -> Coords1d {
        UNIT_POS_Y
    }
}
