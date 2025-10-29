use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::layout::Coords2d;

use crate::components::{
    fragment::FragmentAttention,
    stage::{
        StageAttentionConfig, kv_reuse_attention::KVReuseStageAttention,
        partitioner::AttentionPartitioner, unit::UnitKVReuseStageConfig,
    },
    tile::UnitReducer,
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

    fn coordinates<S: StageAttentionConfig>(#[comptime] _config: S) -> Coords2d {
        todo!()
    }
}
