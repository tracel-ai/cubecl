use cubecl_core::CubeDim;

use crate::components::{AttentionSetupError, global::GlobalConfig, stage::StageConfig};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct DummyGlobalConfig<S: StageConfig> {
    stage_config: S,
    num_planes: u32,
}

impl<S: StageConfig> GlobalConfig for DummyGlobalConfig<S> {
    fn cube_dim(&self) -> CubeDim {
        CubeDim::new_2d(self.plane_dim(), self.num_planes)
    }

    fn plane_dim(&self) -> u32 {
        self.stage_config.plane_dim()
    }
}

impl<S: StageConfig> DummyGlobalConfig<S> {
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
