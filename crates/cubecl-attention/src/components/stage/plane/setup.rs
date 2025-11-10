use std::marker::PhantomData;

use crate::components::{
    attention_types::*,
    stage::{
        AttentionTilingLayout,
        plane::{PlanePartitionAttention, config::PlanePartitionStageConfig},
    },
    tile::TileAttentionFamily,
};
use cubecl_core::{client::ComputeClient, prelude::ReadWrite};
use cubecl_matmul::components::{stage::StageFamily, tile::io::Strided};

use crate::components::{
    AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError, stage::StageAttentionFamily,
};

pub struct PlanePartitionStageAttentionFamily<
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
> StageAttentionFamily for PlanePartitionStageAttentionFamily<FA, SK, SV, SO>
{
    type Attention<AP: AttentionPrecision> = PlanePartitionAttention<
        AP,
        SK::Stage<KS<AP>, AttentionTilingLayout>,
        SV::Stage<VS<AP>, AttentionTilingLayout>,
        SO::Stage<OS<AP>, AttentionTilingLayout>,
        FA::TileAttention<AP>,
    >;

    type KeyStage = SK;
    type ValueStage = SV;
    type OutStage = SO;

    type Config = PlanePartitionStageConfig<FA::Config>;

    fn setup<AP: crate::components::AttentionPrecision, R: cubecl_core::Runtime>(
        client: &ComputeClient<R::Server>,
        problem: &AttentionProblem,
        selection: &AttentionSelection,
        line_sizes: &AttentionLineSizes,
    ) -> Result<Self::Config, AttentionSetupError> {
        let num_planes = selection.tiling_scheme.stage_size.seq_q
            * FA::computation_resources()?.num_planes(selection.plane_dim)?;

        let tile_config = FA::setup::<AP, R>(client, problem, selection, line_sizes, num_planes)?;

        PlanePartitionStageConfig::new(
            tile_config,
            selection.tiling_scheme,
            selection.reuse_key_value,
            num_planes,
        )
    }
}
