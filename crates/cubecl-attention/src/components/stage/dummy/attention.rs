use std::marker::PhantomData;

use crate::components::{
    AttentionPrecision,
    stage::{StageAttention, dummy::config::DummyStageConfig},
    tile::TileAttention,
};

pub struct DummyStageAttention<AP: AttentionPrecision, TA: TileAttention<AP>> {
    _phantom: PhantomData<(AP, TA)>,
}

impl<TA: TileAttention<AP>, AP: AttentionPrecision> StageAttention<AP>
    for DummyStageAttention<AP, TA>
{
    type Config = DummyStageConfig<TA::Config>;
}
