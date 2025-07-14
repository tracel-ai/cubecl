use crate::components::{AttentionSetupError, stage::StageConfig, tile::TileConfig};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct DummyStageConfig<T: TileConfig> {
    tile_config: T,
    num_planes: u32,
}

impl<T: TileConfig> StageConfig for DummyStageConfig<T> {
    fn plane_dim(&self) -> u32 {
        self.tile_config.plane_dim()
    }

    fn num_planes(&self) -> u32 {
        self.num_planes
    }
}

impl<T: TileConfig> DummyStageConfig<T> {
    pub fn new(tile_config: T, num_planes: u32) -> Result<Self, AttentionSetupError> {
        Self {
            tile_config,
            num_planes,
        }
        .validate()
    }

    pub fn validate(self) -> Result<Self, AttentionSetupError> {
        Ok(self)
    }
}
