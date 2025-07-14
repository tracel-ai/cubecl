use std::marker::PhantomData;

use cubecl_core::client::ComputeClient;

use crate::components::{
    AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError,
    global::{
        GlobalAttentionFamily,
        dummy::{DummyGlobalAttention, config::DummyGlobalConfig},
    },
    stage::{StageAttentionFamily, StageConfig as _},
};

pub struct DummyGlobalAttentionFamily<SA: StageAttentionFamily> {
    _phantom: PhantomData<SA>,
}

impl<SA: StageAttentionFamily> GlobalAttentionFamily for DummyGlobalAttentionFamily<SA> {
    type Attention<AP: AttentionPrecision> = DummyGlobalAttention<AP, SA::Attention<AP>>;

    type Config = DummyGlobalConfig<SA::Config>;

    fn setup<AP: crate::components::AttentionPrecision, R: cubecl_core::Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &AttentionProblem,
        selection: &AttentionSelection,
        line_sizes: &AttentionLineSizes,
    ) -> Result<Self::Config, AttentionSetupError> {
        let stage_config = SA::setup::<AP, R>(client, problem, selection, line_sizes)?;

        DummyGlobalConfig::new(stage_config, stage_config.num_planes())
    }
}
