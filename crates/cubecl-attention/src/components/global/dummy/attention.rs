use std::marker::PhantomData;

use crate::components::{
    AttentionPrecision,
    global::{GlobalAttention, dummy::config::DummyGlobalConfig},
    stage::StageAttention,
};

pub struct DummyGlobalAttention<AP: AttentionPrecision, SA: StageAttention<AP>> {
    _phantom: PhantomData<(AP, SA)>,
}

impl<SA: StageAttention<AP>, AP: AttentionPrecision> GlobalAttention<AP>
    for DummyGlobalAttention<AP, SA>
{
    type Config = DummyGlobalConfig<SA::Config>;
}
