use std::marker::PhantomData;

use cubecl_core::client::ComputeClient;
use cubecl_matmul::components::{global::PartitionedStageFamily, stage::StridedStageFamily};

use crate::components::{
    AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError,
    global::{
        GlobalAttentionFamily,
        dummy::{DummyGlobalAttention, config::DummyGlobalConfig},
    },
    stage::{StageAttentionConfig as _, StageAttentionFamily},
};

pub struct DummyGlobalAttentionFamily<SA: StageAttentionFamily> {
    _phantom: PhantomData<SA>,
}

impl<
    SA: StageAttentionFamily<
            KeyStage = StridedStageFamily,
            ValueStage = StridedStageFamily,
            OutStage = PartitionedStageFamily,
        >,
> GlobalAttentionFamily for DummyGlobalAttentionFamily<SA>
{
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
