use crate::components::{
    AttentionPrecision,
    stage::{
        StageAttentionFamily,
        dummy::{DummyStageAttention, config::DummyStageConfig},
    },
};

pub struct DummyStageAttentionFamily {}
impl StageAttentionFamily for DummyStageAttentionFamily {
    type Attention<AP: AttentionPrecision> = DummyStageAttention;

    type Config = DummyStageConfig;

    fn setup<AP: crate::components::AttentionPrecision, R: cubecl_core::Runtime>(
        client: &cubecl_core::prelude::ComputeClient<R::Server, R::Channel>,
        problem: &crate::components::AttentionProblem,
        selection: &crate::components::AttentionSelection,
        line_sizes: &crate::components::AttentionLineSizes,
    ) -> Result<Self::Config, crate::components::AttentionSetupError> {
        todo!()
    }
}
