use std::marker::PhantomData;

use crate::components::{
    AttentionElems,
    attention_types::*,
    stage::{
        AttentionTilingLayout, PartitionAttentionConfig, SharedPartitionAttentionConfig,
        unit::{UnitPartitionAttention, UnitPartitionStageConfig},
        validate,
    },
    tile::TileAttentionFamily,
};
use cubecl_core::{client::ComputeClient, prelude::ReadWrite};
use cubecl_matmul::components::{
    ComputeResources, MatrixLayout,
    stage::{StageFamily, StageMemoryConfig, SwizzleMode},
    tile::io::Strided,
};

use crate::components::{
    AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError, stage::StageAttentionFamily,
};

pub struct UnitPartitionStageAttentionFamily<
    TA: TileAttentionFamily,
    SK: StageFamily,
    SV: StageFamily,
    SO: StageFamily<ReadWrite>,
> {
    _phantom: PhantomData<(TA, SK, SV, SO)>,
}

impl<
    TA: TileAttentionFamily,
    SK: StageFamily<TileKind = Strided>,
    SV: StageFamily<TileKind = Strided>,
    SO: StageFamily<ReadWrite, TileKind = Strided>,
> StageAttentionFamily for UnitPartitionStageAttentionFamily<TA, SK, SV, SO>
{
    type Attention<AP: AttentionPrecision> = UnitPartitionAttention<
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
        client: &ComputeClient<R>,
        problem: &AttentionProblem,
        selection: &AttentionSelection,
        line_sizes: &AttentionLineSizes,
        dtypes: &AttentionElems,
    ) -> Result<Self::Config, AttentionSetupError> {
        let compute_resources = if let ComputeResources::Units(units) = TA::computation_resources()?
        {
            ComputeResources::Units(units * selection.tiling_scheme.stage_size.seq_q)
        } else {
            return Err(AttentionSetupError::InvalidConfig(Box::new(
                "Error: Tried to use a unit stage attention with a plane tile attention."
                    .to_string(),
            )));
        };

        let num_planes = compute_resources.num_planes(selection.plane_dim)?;
        let tile_config = TA::setup(client, problem, selection, line_sizes, num_planes, dtypes)?;

        let key_smem_config = StageMemoryConfig {
            num_planes,
            elements_per_tile_along_row: selection.tiling_scheme.tile_size.seq_kv,
            elements_per_tile_along_col: selection.tiling_scheme.tile_size.head_dim,
            tiles_per_partition_along_row: selection.tiling_scheme.partition_size.seq_kv,
            tiles_per_partition_along_col: selection.tiling_scheme.partition_size.head_dim,
            partitions_per_stage_along_row: 1,
            partitions_per_stage_along_col: 1,
            line_size: line_sizes.key as u32,
            matrix_layout: MatrixLayout::RowMajor,
            swizzle: SwizzleMode::None,
            num_stages: 1,
        };

        let value_smem_config = StageMemoryConfig {
            num_planes,
            elements_per_tile_along_row: selection.tiling_scheme.tile_size.seq_kv,
            elements_per_tile_along_col: selection.tiling_scheme.tile_size.val_dim,
            tiles_per_partition_along_row: selection.tiling_scheme.partition_size.seq_kv,
            tiles_per_partition_along_col: selection.tiling_scheme.partition_size.val_dim,
            partitions_per_stage_along_row: 1,
            partitions_per_stage_along_col: 1,
            line_size: line_sizes.value as u32,
            matrix_layout: MatrixLayout::RowMajor,
            swizzle: SwizzleMode::None,
            num_stages: 1,
        };

        let out_smem_config = StageMemoryConfig {
            num_planes,
            elements_per_tile_along_row: selection.tiling_scheme.tile_size.seq_q,
            elements_per_tile_along_col: selection.tiling_scheme.tile_size.val_dim,
            tiles_per_partition_along_row: 1,
            tiles_per_partition_along_col: 1,
            // Each unit has its slot in row direction
            partitions_per_stage_along_row: num_planes * selection.plane_dim,
            partitions_per_stage_along_col: 1,
            line_size: line_sizes.out as u32,
            matrix_layout: MatrixLayout::RowMajor,
            swizzle: SwizzleMode::None,
            num_stages: 1,
        };

        validate(PartitionAttentionConfig::Unit(UnitPartitionStageConfig {
            shared: SharedPartitionAttentionConfig {
                tile_config,
                partition_size: selection.tiling_scheme.partition_size,
                stage_size: selection.tiling_scheme.stage_size,
                reuse_key_value: selection.reuse_key_value,
                num_planes,
                key_smem_config,
                value_smem_config,
                out_smem_config,
            },
        }))
    }
}
