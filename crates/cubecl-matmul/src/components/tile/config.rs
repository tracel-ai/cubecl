use crate::components::TileSize;
use std::{fmt::Debug, hash::Hash};

// This serves as interface for higher level matmuls, not for what is used within tile matmul
pub trait TileConfig: Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static {
    /// Returns the line size for the given ident
    fn plane_dim(&self) -> u32;
}

/// Configuration for the Tile Matmul level
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct SharedTileConfig {
    pub tile_size: TileSize,
    pub plane_dim: u32,
}

impl TileConfig for SharedTileConfig {
    fn plane_dim(&self) -> u32 {
        self.plane_dim
    }
}

impl SharedTileConfig {
    pub fn new(tile_size: TileSize, plane_dim: u32) -> Self {
        SharedTileConfig {
            tile_size,
            plane_dim,
        }
    }
}
