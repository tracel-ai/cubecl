use std::marker::PhantomData;

use crate::components::{
    attention_types::*,
    stage::{
        AttentionTilingLayout,
        unit::{UnitPartitionAttention, config::UnitPartitionStageConfig},
    },
    tile::TileAttentionFamily,
};
use cubecl_core::{client::ComputeClient, prelude::ReadWrite};
use cubecl_matmul::components::{ComputeResources, stage::StageFamily, tile::io::Strided};

use crate::components::{
    AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError, stage::StageAttentionFamily,
};

pub struct UnitPartitionStageAttentionFamily<
    FA: TileAttentionFamily,
    SK: StageFamily,
    SV: StageFamily,
    SO: StageFamily<ReadWrite>,
> {
    _phantom: PhantomData<(FA, SK, SV, SO)>,
}

impl<
    FA: TileAttentionFamily,
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
        FA::TileAttention<AP>,
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
                "Error: Tried to use a unit stage attention with a plane tile attention."
                    .to_string(),
            )));
        };

        let num_planes = compute_resources.num_planes(selection.plane_dim)?;
        let tile_config = FA::setup::<AP, R>(client, problem, selection, line_sizes, num_planes)?;

        UnitPartitionStageConfig::new(
            tile_config,
            selection.tiling_scheme,
            selection.reuse_key_value,
            num_planes,
        )
    }
}
