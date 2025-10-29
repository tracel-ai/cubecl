use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::layout::Coords2d;

use crate::components::{
    fragment::FragmentAttention,
    stage::{
        StageAttentionConfig, kv_reuse_attention::KVReuseStageAttention,
        partitioner::AttentionPartitioner, plane::PlaneKVReuseStageConfig,
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

    fn coordinates<S: StageAttentionConfig>(#[comptime] _config: S) -> Coords2d {
        todo!()
    }
}
