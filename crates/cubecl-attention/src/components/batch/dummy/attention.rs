use std::marker::PhantomData;

use crate::components::{
    AttentionPrecision,
    batch::{BatchAttention, dummy::config::DummyBatchConfig},
    global::GlobalAttention,
};

pub struct DummyBatchAttention<AP: AttentionPrecision, GA: GlobalAttention<AP>> {
    _phantom: PhantomData<(AP, GA)>,
}

impl<GA: GlobalAttention<AP>, AP: AttentionPrecision> BatchAttention<AP>
    for DummyBatchAttention<AP, GA>
{
    type Config = DummyBatchConfig<GA::Config>;
}
