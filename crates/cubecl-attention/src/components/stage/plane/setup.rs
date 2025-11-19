use std::marker::PhantomData;

use crate::components::{
    AttentionElems,
    attention_types::*,
    stage::{
        AttentionTilingLayout, PartitionAttentionConfig, SharedPartitionAttentionConfig,
        plane::{PlanePartitionAttention, config::PlanePartitionStageConfig},
        validate,
    },
    tile::{TileAttentionConfig, TileAttentionFamily},
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
    TA: TileAttentionFamily,
    SK: StageFamily<TileKind = Strided>,
    SV: StageFamily<TileKind = Strided>,
    SO: StageFamily<ReadWrite, TileKind = Strided>,
> StageAttentionFamily for PlanePartitionStageAttentionFamily<TA, SK, SV, SO>
{
    type Attention<AP: AttentionPrecision> = PlanePartitionAttention<
        AP,
        SK::Stage<KS<AP>, AttentionTilingLayout>,
        SV::Stage<VS<AP>, AttentionTilingLayout>,
        SO::Stage<OS<AP>, AttentionTilingLayout>,
        TA::TileAttention<AP>,
    >;

    type KeyStage = SK;
    type ValueStage = SV;
    type OutStage = SO;

    type Config = PartitionAttentionConfig<TA::Config>;

    fn setup<R: cubecl_core::Runtime>(
        client: &ComputeClient<R::Server>,
        problem: &AttentionProblem,
        selection: &AttentionSelection,
        line_sizes: &AttentionLineSizes,
        dtypes: &AttentionElems,
    ) -> Result<Self::Config, AttentionSetupError> {
        let num_planes = selection.tiling_scheme.stage_size.seq_q
            * TA::computation_resources()?.num_planes(selection.plane_dim)?;

        let tile_config =
            TA::setup::<R>(client, problem, selection, line_sizes, num_planes, dtypes)?;

        validate(PartitionAttentionConfig::Plane(PlanePartitionStageConfig {
            shared: SharedPartitionAttentionConfig {
                tile_config,
                partition_size: selection.tiling_scheme.partition_size,
                reuse_key_value: selection.reuse_key_value,
                num_planes,
            },
        }))
    }
}
