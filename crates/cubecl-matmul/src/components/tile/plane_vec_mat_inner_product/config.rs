use crate::components::tile::{SharedTileConfig, TileConfig};

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct PlaneVecMatInnerProductConfig {
    pub shared: SharedTileConfig,
    pub reduce_line_size: u32,
}

impl PlaneVecMatInnerProductConfig {
    pub fn new(shared: SharedTileConfig, reduce_line_size: u32) -> Self {
        Self {
            shared,
            reduce_line_size,
        }
    }
}

impl TileConfig for PlaneVecMatInnerProductConfig {
    fn plane_dim(&self) -> u32 {
        self.shared.plane_dim()
    }
}
