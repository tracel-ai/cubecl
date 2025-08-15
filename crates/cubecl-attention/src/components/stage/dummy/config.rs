use crate::components::{
    AttentionSetupError,
    stage::StageAttentionConfig,
    tile::{TileAttentionConfig, dummy::AttentionStageMemoryConfig},
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct DummyStageConfig<TC: TileAttentionConfig> {
    tile_config: TC,
    score_stage_memory_config: AttentionStageMemoryConfig<TC::ScoreConfig>,
    value_stage_memory_config: AttentionStageMemoryConfig<TC::ValueConfig>,
}

impl<TC: TileAttentionConfig> StageAttentionConfig for DummyStageConfig<TC> {
    type TileAttentionConfig = TC;
    type ScoreStageMemoryConfig = AttentionStageMemoryConfig<TC::ScoreConfig>;
    type ValueStageMemoryConfig = AttentionStageMemoryConfig<TC::ValueConfig>;

    fn plane_dim(&self) -> u32 {
        32
    }

    fn num_planes(&self) -> u32 {
        1
    }

    fn rows_per_plane(&self) -> u32 {
        1
    }

    fn tile_config(&self) -> Self::TileAttentionConfig {
        self.tile_config
    }

    fn score_stage_memory_config(&self) -> Self::ScoreStageMemoryConfig {
        self.score_stage_memory_config
    }

    fn value_stage_memory_config(&self) -> Self::ValueStageMemoryConfig {
        self.value_stage_memory_config
    }
}

impl<TC: TileAttentionConfig> DummyStageConfig<TC> {
    pub fn new(
        tile_config: TC,
        score_stage_memory_config: AttentionStageMemoryConfig<TC::ScoreConfig>,
        value_stage_memory_config: AttentionStageMemoryConfig<TC::ValueConfig>,
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
