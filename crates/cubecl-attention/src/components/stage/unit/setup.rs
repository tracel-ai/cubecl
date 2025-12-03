use std::marker::PhantomData;

use crate::components::{
    attention_types::*,
    stage::{
        AttentionTilingLayout, PartitionAttentionConfig, SharedPartitionAttentionConfig,
        unit::{UnitPartitionAttention, UnitPartitionStageConfig},
        validate,
    },
    tile::TileAttentionFamily,
};
use cubecl_core::prelude::ReadWrite;
use cubecl_matmul::components::{
    MatrixLayout,
    stage::{StageFamily, StageMemoryConfig, SwizzleMode},
    tile::io::Strided,
};

use crate::components::{
    AttentionBlueprint, AttentionPrecision, AttentionSetupError, stage::StageAttentionFamily,
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

    fn expand_blueprint(
        blueprint: &AttentionBlueprint,
    ) -> Result<Self::Config, AttentionSetupError> {
        let tile_config = TA::expand_blueprint(blueprint)?;

        let key_smem_config = StageMemoryConfig {
            num_planes: blueprint.num_planes,
            elements_per_tile_along_row: blueprint.tiling_scheme.tile_size.seq_kv,
            elements_per_tile_along_col: blueprint.tiling_scheme.tile_size.head_dim,
            tiles_per_partition_along_row: blueprint.tiling_scheme.partition_size.seq_kv,
            tiles_per_partition_along_col: blueprint.tiling_scheme.partition_size.head_dim,
            partitions_per_stage_along_row: 1,
            partitions_per_stage_along_col: 1,
            line_size: blueprint.line_sizes.key as u32,
            matrix_layout: MatrixLayout::RowMajor,
            swizzle: SwizzleMode::None,
            num_stages: 1,
        };

        let value_smem_config = StageMemoryConfig {
            num_planes: blueprint.num_planes,
            elements_per_tile_along_row: blueprint.tiling_scheme.tile_size.seq_kv,
            elements_per_tile_along_col: blueprint.tiling_scheme.tile_size.val_dim,
            tiles_per_partition_along_row: blueprint.tiling_scheme.partition_size.seq_kv,
            tiles_per_partition_along_col: blueprint.tiling_scheme.partition_size.val_dim,
            partitions_per_stage_along_row: 1,
            partitions_per_stage_along_col: 1,
            line_size: blueprint.line_sizes.value as u32,
            matrix_layout: MatrixLayout::RowMajor,
            swizzle: SwizzleMode::None,
            num_stages: 1,
        };

        let out_smem_config = StageMemoryConfig {
            num_planes: blueprint.num_planes,
            elements_per_tile_along_row: blueprint.tiling_scheme.tile_size.seq_q,
            elements_per_tile_along_col: blueprint.tiling_scheme.tile_size.val_dim,
            tiles_per_partition_along_row: 1,
            tiles_per_partition_along_col: 1,
            // Each unit has its slot in row direction
            partitions_per_stage_along_row: blueprint.num_planes * blueprint.plane_dim,
            partitions_per_stage_along_col: 1,
            line_size: blueprint.line_sizes.out as u32,
            matrix_layout: MatrixLayout::RowMajor,
            swizzle: SwizzleMode::None,
            num_stages: 1,
        };

        validate(PartitionAttentionConfig::Unit(UnitPartitionStageConfig {
            shared: SharedPartitionAttentionConfig {
                tile_config,
                partition_size: blueprint.tiling_scheme.partition_size,
                stage_size: blueprint.tiling_scheme.stage_size,
                reuse_key_value: blueprint.reuse_key_value,
                num_planes: blueprint.num_planes,
                key_smem_config,
                value_smem_config,
                out_smem_config,
            },
        }))
    }
}
