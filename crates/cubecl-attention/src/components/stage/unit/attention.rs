use crate::components::{
    fragment::FragmentAttention,
    stage::{kv_reuse_attention::KVReuseStageAttention, unit::config::UnitKVReuseStageConfig},
};

pub type UnitKVReuseStageAttention<AP, SK, SV, SO, FA> = KVReuseStageAttention<
    AP,
    SK,
    SV,
    SO,
    FA,
    UnitKVReuseStageConfig<<FA as FragmentAttention<AP>>::Config>,
>;
