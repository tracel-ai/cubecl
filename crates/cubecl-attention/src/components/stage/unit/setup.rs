use std::marker::PhantomData;

use crate::components::{
    attention_types::*,
    stage::{
        AttentionStageMemoryConfig, AttentionTilingLayout,
        unit::{UnitPartitionAttention, config::UnitPartitionStageConfig},
    },
    tile::FragmentAttentionFamily,
};
use cubecl_core::{client::ComputeClient, prelude::ReadWrite};
use cubecl_matmul::components::{
    ComputeResources, GlobalPartitionSize, TilingScheme, stage::StageFamily, tile::io::Strided,
};

use crate::components::{
    AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError, stage::StageAttentionFamily,
};

pub struct UnitPartitionStageAttentionFamily<
    FA: FragmentAttentionFamily,
    SK: StageFamily,
    SV: StageFamily,
    SO: StageFamily<ReadWrite>,
> {
    _phantom: PhantomData<(FA, SK, SV, SO)>,
}

impl<
    FA: FragmentAttentionFamily,
    SK: StageFamily<TileKind = Strided>,
    SV: StageFamily<TileKind = Strided>,
    SO: StageFamily<ReadWrite, TileKind = Strided>,
> StageAttentionFamily for UnitPartitionStageAttentionFamily<FA, SK, SV, SO>
{
    type Attention<AP: AttentionPrecision> = UnitPartitionAttention<
        AP,
        SK::Stage<KS<AP>, AttentionTilingLayout>,
        SV::Stage<VS<AP>, AttentionTilingLayout>,
        SO::Stage<OS<AP>, AttentionTilingLayout>,
        FA::FragmentAttention<AP>,
    >;

    type KeyStage = SK;
    type ValueStage = SV;
    type OutStage = SO;

    type Config = UnitPartitionStageConfig<FA::Config>;

    fn setup<AP: crate::components::AttentionPrecision, R: cubecl_core::Runtime>(
        client: &ComputeClient<R::Server>,
        problem: &AttentionProblem,
        selection: &AttentionSelection,
        line_sizes: &AttentionLineSizes,
    ) -> Result<Self::Config, AttentionSetupError> {
        let compute_resources = if let ComputeResources::Units(units) = FA::computation_resources()?
        {
            ComputeResources::Units(units * selection.tiling_scheme.stage_size.seq_q)
        } else {
            return Err(AttentionSetupError::InvalidConfig(Box::new(
                "Error: Tried to use a unit stage attention with a plane fragment attention."
                    .to_string(),
            )));
        };

        let num_planes = compute_resources.num_planes(selection.plane_dim)?;
        let tile_config = FA::setup::<AP, R>(client, problem, selection, line_sizes, num_planes)?;

        UnitPartitionStageConfig::new(
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
