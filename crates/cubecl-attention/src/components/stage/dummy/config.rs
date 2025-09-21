use cubecl_matmul::components::{MatrixLayout, StageIdent, TilingScheme, stage::StageMemoryConfig};

use crate::components::{
    AttentionSetupError, AttentionTilingScheme, stage::StageAttentionConfig,
    tile::dummy::FlashMatmulConfig,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct DummyStageConfig<FC: FlashMatmulConfig> {
    tile_config: FC,
    score_stage_memory_config: AttentionStageMemoryConfig,
    value_stage_memory_config: AttentionStageMemoryConfig,
    tiling_scheme: AttentionTilingScheme,
    reuse_key_value: bool,
}

impl<FC: FlashMatmulConfig> StageAttentionConfig for DummyStageConfig<FC> {
    type FlashMatmulConfig = FC;

    fn plane_dim(&self) -> u32 {
        32
    }

    fn num_planes(&self) -> u32 {
        // TODO increase with stage_seq_q > 1
        1
    }

    fn tile_config(&self) -> Self::FlashMatmulConfig {
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
}

impl<FC: FlashMatmulConfig> DummyStageConfig<FC> {
    pub fn new(
        tile_config: FC,
        score_stage_memory_config: AttentionStageMemoryConfig,
        value_stage_memory_config: AttentionStageMemoryConfig,
        tiling_scheme: AttentionTilingScheme,
        reuse_key_value: bool,
    ) -> Result<Self, AttentionSetupError> {
        Self {
            tile_config,
            score_stage_memory_config,
            value_stage_memory_config,
            tiling_scheme,
            reuse_key_value,
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

impl StageMemoryConfig for AttentionStageMemoryConfig {
    fn num_main_flow_planes(&self) -> u32 {
        // TODO increase with stage_seq_q > 1
        1
    }

    fn tiling_scheme(&self) -> TilingScheme {
        self.matmul_tiling_scheme
    }

    fn stage_line_size(&self, _ident: StageIdent) -> u32 {
        1
    }

    fn matrix_layout(&self, _ident: StageIdent) -> MatrixLayout {
        MatrixLayout::RowMajor
    }

    fn num_stages(&self, _ident: StageIdent) -> u32 {
        1
    }
}
