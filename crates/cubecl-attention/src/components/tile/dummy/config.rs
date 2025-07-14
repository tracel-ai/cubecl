use crate::components::{AttentionSetupError, tile::TileConfig};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct DummyTileConfig {
    plane_dim: u32,
}

impl TileConfig for DummyTileConfig {
    fn plane_dim(&self) -> u32 {
        self.plane_dim
    }
}

impl DummyTileConfig {
    pub fn new(plane_dim: u32) -> Result<Self, AttentionSetupError> {
        Self { plane_dim }.validate()
    }

    pub fn validate(self) -> Result<Self, AttentionSetupError> {
        Ok(self)
    }
}
