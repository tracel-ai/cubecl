use crate::components::{
    AttentionPrecision,
    global::{
        GlobalAttentionFamily,
        dummy::{DummyGlobalAttention, config::DummyGlobalConfig},
    },
};

pub struct DummyGlobalAttentionFamily {}
impl GlobalAttentionFamily for DummyGlobalAttentionFamily {
    type Attention<AP: AttentionPrecision> = DummyGlobalAttention;

    type Config = DummyGlobalConfig;

    fn setup<AP: crate::components::AttentionPrecision, R: cubecl_core::Runtime>(
        client: &cubecl_core::prelude::ComputeClient<R::Server, R::Channel>,
        problem: &crate::components::AttentionProblem,
        selection: &crate::components::AttentionSelection,
        line_sizes: &crate::components::AttentionLineSizes,
    ) -> Result<Self::Config, crate::components::AttentionSetupError> {
        todo!()
    }
}
