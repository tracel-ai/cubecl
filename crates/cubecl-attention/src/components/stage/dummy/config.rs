use cubecl_matmul::components::tile::TileConfig;

use crate::components::{AttentionSetupError, stage::StageConfig};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct DummyStageConfig<ST: TileConfig, VT: TileConfig> {
    score_tile_config: ST,
    value_tile_config: VT,
    num_planes: u32,
}

impl<ST: TileConfig, VT: TileConfig> StageConfig for DummyStageConfig<ST, VT> {
    fn plane_dim(&self) -> u32 {
        self.score_tile_config.plane_dim()
    }

    fn num_planes(&self) -> u32 {
        self.num_planes
    }

    fn rows_per_plane(&self) -> u32 {
        // self.tiling_scheme...
        8
    }
}

impl<ST: TileConfig, VT: TileConfig> DummyStageConfig<ST, VT> {
    pub fn new(
        score_tile_config: ST,
        value_tile_config: VT,
        num_planes: u32,
    ) -> Result<Self, AttentionSetupError> {
        Self {
            score_tile_config,
            value_tile_config,
            num_planes,
        }
        .validate()
    }

    pub fn validate(self) -> Result<Self, AttentionSetupError> {
        Ok(self)
    }
}
