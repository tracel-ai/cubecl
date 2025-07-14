use crate::components::{
    AttentionPrecision,
    batch::{BatchAttention, dummy::config::DummyBatchConfig},
};

pub struct DummyBatchAttention {}

impl<AP: AttentionPrecision> BatchAttention<AP> for DummyBatchAttention {
    type Config = DummyBatchConfig;
}
