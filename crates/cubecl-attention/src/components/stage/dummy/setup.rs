use std::marker::PhantomData;

use crate::components::{
    AttentionPrecision,
    stage::{
        StageAttentionFamily,
        dummy::{DummyStageAttention, config::DummyStageConfig},
    },
    tile::TileAttentionFamily,
};

pub struct DummyStageAttentionFamily<TA: TileAttentionFamily> {
    _phantom: PhantomData<TA>,
}

impl<TA: TileAttentionFamily> StageAttentionFamily for DummyStageAttentionFamily<TA> {
    type Attention<AP: AttentionPrecision> = DummyStageAttention<AP, TA::Attention<AP>>;

    type Config = DummyStageConfig<TA::Config>;

    fn setup<AP: crate::components::AttentionPrecision, R: cubecl_core::Runtime>(
        client: &cubecl_core::prelude::ComputeClient<R::Server, R::Channel>,
        problem: &crate::components::AttentionProblem,
        selection: &crate::components::AttentionSelection,
        line_sizes: &crate::components::AttentionLineSizes,
    ) -> Result<Self::Config, crate::components::AttentionSetupError> {
        let tile_config = TA::setup::<AP, R>(client, problem, selection, line_sizes)?;

        DummyStageConfig::new(tile_config, 1)
    }
}
