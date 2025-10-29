use crate::components::{
    fragment::FragmentAttention,
    stage::{kv_reuse_attention::KVReuseStageAttention, plane::config::PlaneKVReuseStageConfig},
};

pub type PlaneKVReuseStageAttention<AP, SK, SV, SO, FA> = KVReuseStageAttention<
    AP,
    SK,
    SV,
    SO,
    FA,
    PlaneKVReuseStageConfig<<FA as FragmentAttention<AP>>::Config>,
>;
