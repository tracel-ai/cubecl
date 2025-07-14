use crate::components::{
    AttentionPrecision,
    global::{GlobalAttention, dummy::config::DummyGlobalConfig},
};

pub struct DummyGlobalAttention {}

impl<AP: AttentionPrecision> GlobalAttention<AP> for DummyGlobalAttention {
    type Config = DummyGlobalConfig;
}
