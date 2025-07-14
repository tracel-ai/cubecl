use crate::components::{
    AttentionPrecision,
    stage::{StageAttention, dummy::config::DummyStageConfig},
};

pub struct DummyStageAttention {}

impl<AP: AttentionPrecision> StageAttention<AP> for DummyStageAttention {
    type Config = DummyStageConfig;
}
