use std::marker::PhantomData;

use crate::components::{attention_types::*, fragment::AttentionMatmulFamily};
use cubecl_core::{client::ComputeClient, prelude::ReadWrite};
use cubecl_matmul::components::{
    GlobalPartitionSize, TilingScheme, stage::StageFamily, tile::io::Strided,
};

use crate::components::{
    AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError,
    stage::{
        StageAttentionFamily,
        simple_kv_reuse::{
            AttentionStageMemoryConfig, SimpleKVReuseStageAttention, SimpleKVReuseStageConfig,
        },
    },
    tile::AttentionTilingLayout,
};

pub struct SimpleKVReuseStageAttentionFamily<
    AM: AttentionMatmulFamily,
    SK: StageFamily,
    SV: StageFamily,
    SO: StageFamily<ReadWrite>,
> {
    _phantom: PhantomData<(AM, SK, SV, SO)>,
}

impl<
    AM: AttentionMatmulFamily,
    SK: StageFamily<TileKind = Strided>,
    SV: StageFamily<TileKind = Strided>,
    SO: StageFamily<ReadWrite, TileKind = Strided>,
> StageAttentionFamily for SimpleKVReuseStageAttentionFamily<AM, SK, SV, SO>
{
    type Attention<AP: AttentionPrecision> = SimpleKVReuseStageAttention<
        AP,
        SK::Stage<KS<AP>, AttentionTilingLayout>,
        SV::Stage<VS<AP>, AttentionTilingLayout>,
        SO::Stage<OS<AP>, AttentionTilingLayout>,
        AM::Matmul<AP>,
    >;

    type KeyStage = SK;
    type ValueStage = SV;
    type OutStage = SO;

    type Config = SimpleKVReuseStageConfig<AM::Config>;

    fn setup<AP: crate::components::AttentionPrecision, R: cubecl_core::Runtime>(
        client: &ComputeClient<R::Server>,
        problem: &AttentionProblem,
        selection: &AttentionSelection,
        line_sizes: &AttentionLineSizes,
    ) -> Result<Self::Config, AttentionSetupError> {
        let num_planes = selection.tiling_scheme.stage_size.seq_q
            * AM::computation_resources()?.num_planes(selection.plane_dim)?;

        let tile_config = AM::setup::<AP, R>(client, problem, selection, line_sizes, num_planes)?;

        SimpleKVReuseStageConfig::new(
            tile_config,
            score_attention_stage_memory_config(selection),
            value_attention_stage_memory_config(selection),
            selection.tiling_scheme,
            selection.reuse_key_value,
            num_planes,
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
