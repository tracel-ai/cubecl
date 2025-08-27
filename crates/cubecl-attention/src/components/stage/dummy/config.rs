use crate::components::{
    AttentionSetupError,
    stage::StageAttentionConfig,
    tile::dummy::{AttentionStageMemoryConfig, FlashMatmulConfig},
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct DummyStageConfig<FC: FlashMatmulConfig> {
    tile_config: FC,
    score_stage_memory_config: AttentionStageMemoryConfig<FC::ScoreConfig>,
    value_stage_memory_config: AttentionStageMemoryConfig<FC::ValueConfig>,
}

impl<FC: FlashMatmulConfig> StageAttentionConfig for DummyStageConfig<FC> {
    type FlashMatmulConfig = FC;
    type ScoreStageMemoryConfig = AttentionStageMemoryConfig<FC::ScoreConfig>;
    type ValueStageMemoryConfig = AttentionStageMemoryConfig<FC::ValueConfig>;

    fn plane_dim(&self) -> u32 {
        32
    }

    fn num_planes(&self) -> u32 {
        1
    }

    fn tile_config(&self) -> Self::FlashMatmulConfig {
        self.tile_config
    }

    fn score_stage_memory_config(&self) -> Self::ScoreStageMemoryConfig {
        self.score_stage_memory_config
    }

    fn value_stage_memory_config(&self) -> Self::ValueStageMemoryConfig {
        self.value_stage_memory_config
    }
}

impl<FC: FlashMatmulConfig> DummyStageConfig<FC> {
    pub fn new(
        tile_config: FC,
        score_stage_memory_config: AttentionStageMemoryConfig<FC::ScoreConfig>,
        value_stage_memory_config: AttentionStageMemoryConfig<FC::ValueConfig>,
    ) -> Result<Self, AttentionSetupError> {
        Self {
            tile_config,
            score_stage_memory_config,
            value_stage_memory_config,
        }
        .validate()
    }

    pub fn validate(self) -> Result<Self, AttentionSetupError> {
        Ok(self)
    }
}
