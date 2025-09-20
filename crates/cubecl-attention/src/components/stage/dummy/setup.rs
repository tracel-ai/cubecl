use std::marker::PhantomData;

use cubecl_core::client::ComputeClient;
use cubecl_matmul::components::{
    GlobalPartitionSize, TilingScheme, stage::StageReaderFamily, tile::loader::Strided,
};

use crate::components::{
    AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError,
    stage::{
        StageAttentionFamily,
        dummy::{AttentionStageMemoryConfig, DummyStageAttention, DummyStageConfig},
    },
    tile::{AttentionTilingLayout, TileAttentionFamily},
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
            score_attention_stage_memory_config(selection),
            value_attention_stage_memory_config(selection),
            selection.tiling_scheme,
            selection.reuse_key_value,
        )
    }
}

fn score_attention_stage_memory_config(
    selection: &AttentionSelection,
) -> AttentionStageMemoryConfig {
    let att_tile_size = selection.tiling_scheme.tile_size;
    let att_partition_size = selection.tiling_scheme.partition_size;
    let att_stage_size = selection.tiling_scheme.stage_size;

    let matmul_tiling_scheme = TilingScheme {
        tile_size: att_tile_size.to_score_matmul_tile_size(),
        partition_size: att_partition_size.to_score_matmul_partition_size(),
        stage_size: (att_stage_size.seq_q, 1, 1).into(),
        global_partition_size: GlobalPartitionSize::new(1, 1, 1),
    };
    AttentionStageMemoryConfig {
        matmul_tiling_scheme,
    }
}

fn value_attention_stage_memory_config(
    selection: &AttentionSelection,
) -> AttentionStageMemoryConfig {
    let att_tile_size = selection.tiling_scheme.tile_size;
    let att_partition_size = selection.tiling_scheme.partition_size;
    let att_stage_size = selection.tiling_scheme.stage_size;

    let matmul_tiling_scheme = TilingScheme {
        tile_size: att_tile_size.to_value_matmul_tile_size(),
        partition_size: att_partition_size.to_value_matmul_partition_size(),
        stage_size: (att_stage_size.seq_q, 1, 1).into(),
        global_partition_size: GlobalPartitionSize::new(1, 1, 1),
    };
    AttentionStageMemoryConfig {
        matmul_tiling_scheme,
    }
}
