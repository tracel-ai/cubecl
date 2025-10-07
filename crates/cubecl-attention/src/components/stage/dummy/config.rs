use cubecl_matmul::components::{MatrixLayout, StageIdent, TilingScheme, stage::StageMemoryConfig};

use crate::components::{
    AttentionSetupError, AttentionTilingScheme, stage::StageAttentionConfig,
    tile::dummy::AttentionMatmulConfig,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct DummyStageConfig<FC: AttentionMatmulConfig> {
    tile_config: FC,
    score_stage_memory_config: AttentionStageMemoryConfig,
    value_stage_memory_config: AttentionStageMemoryConfig,
    tiling_scheme: AttentionTilingScheme,
    reuse_key_value: bool,
    num_planes: u32,
}

impl<FC: AttentionMatmulConfig> StageAttentionConfig for DummyStageConfig<FC> {
    type AttentionMatmulConfig = FC;

    fn plane_dim(&self) -> u32 {
        32
    }

    fn num_planes(&self) -> u32 {
        self.num_planes
    }

    fn tile_config(&self) -> Self::AttentionMatmulConfig {
        self.tile_config
    }

    fn score_stage_memory_config(&self) -> AttentionStageMemoryConfig {
        self.score_stage_memory_config
    }

    fn value_stage_memory_config(&self) -> AttentionStageMemoryConfig {
        self.value_stage_memory_config
    }

    fn tiling_scheme(&self) -> AttentionTilingScheme {
        self.tiling_scheme
    }

    fn reuse_key_value(&self) -> bool {
        self.reuse_key_value
    }

    fn num_rows_per_unit(&self) -> u32 {
        self.tile_config.num_rows_per_unit()
    }
}

impl<FC: AttentionMatmulConfig> DummyStageConfig<FC> {
    pub fn new(
        tile_config: FC,
        score_stage_memory_config: AttentionStageMemoryConfig,
        value_stage_memory_config: AttentionStageMemoryConfig,
        tiling_scheme: AttentionTilingScheme,
        reuse_key_value: bool,
        num_planes: u32,
    ) -> Result<Self, AttentionSetupError> {
        Self {
            tile_config,
            score_stage_memory_config,
            value_stage_memory_config,
            tiling_scheme,
            reuse_key_value,
            num_planes,
        }
        .validate()
    }

    pub fn validate(self) -> Result<Self, AttentionSetupError> {
        if self.reuse_key_value
            && (self.tiling_scheme.tile_size.head_dim != self.tiling_scheme.tile_size.val_dim
                || self.tiling_scheme.partition_size.head_dim
                    != self.tiling_scheme.partition_size.val_dim)
        {
            return Err(AttentionSetupError::InvalidConfig(Box::new(
        "When reusing key/value, head_dim must equal val_dim in both tile_size and partition_size."
            .to_string(),
    )));
        }

        Ok(self)
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct AttentionStageMemoryConfig {
    pub matmul_tiling_scheme: TilingScheme,
}

impl AttentionStageMemoryConfig {
    pub fn into_matmul_config(&self, ident: StageIdent) -> StageMemoryConfig {
        StageMemoryConfig {
            num_main_flow_planes: 1,
            elements_in_tile_row: self.matmul_tiling_scheme.elements_in_tile_row(ident),
            elements_in_tile_col: self.matmul_tiling_scheme.elements_in_tile_col(ident),
            tiles_in_stage_row: self.matmul_tiling_scheme.tiles_in_stage_row(ident),
            tiles_in_stage_col: self.matmul_tiling_scheme.tiles_in_stage_col(ident),
            stage_line_size: 1,
            matrix_layout: MatrixLayout::RowMajor,
            num_stages: 1,
        }
    }
}
