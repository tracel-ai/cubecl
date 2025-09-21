use std::marker::PhantomData;

use cubecl_core::client::ComputeClient;
use cubecl_matmul::components::{stage::StageReaderFamily, tile::reader::Strided};

use crate::components::{
    AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError,
    stage::{
        StageAttentionFamily,
        dummy::{DummyStageAttention, DummyStageConfig},
    },
    tile::{
        AttentionTilingLayout, TileAttentionFamily,
        dummy::{AttentionStageMemoryConfig, FlashMatmulConfig as _},
    },
};

pub struct DummyStageAttentionFamily<TA: TileAttentionFamily, RF: StageReaderFamily> {
    _phantom: PhantomData<(TA, RF)>,
}

impl<TA: TileAttentionFamily, RF: StageReaderFamily<TileKind = Strided>> StageAttentionFamily
    for DummyStageAttentionFamily<TA, RF>
{
    type Attention<AP: AttentionPrecision> =
        DummyStageAttention<AP, RF::Reader<AP::ES, AttentionTilingLayout>, TA::Attention<AP>>;

    type KeyReader = RF;
    type ValueReader = RF;

    type Config = DummyStageConfig<TA::Config>;

    fn setup<AP: crate::components::AttentionPrecision, R: cubecl_core::Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &AttentionProblem,
        selection: &AttentionSelection,
        line_sizes: &AttentionLineSizes,
    ) -> Result<Self::Config, AttentionSetupError> {
        let tile_config = TA::setup::<AP, R>(client, problem, selection, line_sizes)?;

        DummyStageConfig::new(
            tile_config,
            AttentionStageMemoryConfig::new(tile_config.score_config()),
            AttentionStageMemoryConfig::new(tile_config.value_config()),
        )
    }
}
