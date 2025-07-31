use cubecl_core::CubeDim;

use crate::components::{
    AttentionSetupError, global::GlobalAttentionConfig, stage::StageAttentionConfig,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct DummyGlobalConfig<S: StageAttentionConfig> {
    stage_config: S,
    num_planes: u32,
}

impl<S: StageAttentionConfig> GlobalAttentionConfig for DummyGlobalConfig<S> {
    type StageConfig = S;
    type ScoreStageMemoryConfig = S::ScoreStageMemoryConfig;
    type ValueStageMemoryConfig = S::ValueStageMemoryConfig;

    fn score_stage_memory_config(&self) -> Self::ScoreStageMemoryConfig {
        self.stage_config.score_stage_memory_config()
    }

    fn value_stage_memory_config(&self) -> Self::ValueStageMemoryConfig {
        self.stage_config.value_stage_memory_config()
    }

    fn stage_config(&self) -> S {
        self.stage_config
    }

    fn cube_dim(&self) -> CubeDim {
        CubeDim::new_2d(self.plane_dim(), self.num_planes)
    }

    fn plane_dim(&self) -> u32 {
        self.stage_config.plane_dim()
    }

    fn tc(&self) -> u32 {
        // Number of stage iterations = ceil(N/Bc)
        1
    }
}

impl<S: StageAttentionConfig> DummyGlobalConfig<S> {
    pub fn new(stage_config: S, num_planes: u32) -> Result<Self, AttentionSetupError> {
        Self {
            stage_config,
            num_planes,
        }
        .validate()
    }

    pub fn validate(self) -> Result<Self, AttentionSetupError> {
        Ok(self)
    }
}
