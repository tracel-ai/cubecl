use crate::components::{StageIdent, SwizzleConfig, TileSize, stage::SwizzleMode};
use std::{fmt::Debug, hash::Hash};

// This serves as interface for higher level matmuls, not for what is used within tile matmul
pub trait TileConfig: Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static {
    /// Returns the line size for the given ident
    fn plane_dim(&self) -> u32;

    fn elements_in_tile_m(&self) -> u32;

    fn elements_in_tile_n(&self) -> u32;

    fn elements_in_tile_k(&self) -> u32;

    /// Returns the [SwizzleMode] for the given ident
    fn swizzle_mode(&self, ident: StageIdent) -> SwizzleMode;
}

/// Configuration for the Tile Matmul level
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct SharedTileConfig {
    pub tile_size: TileSize,
    pub plane_dim: u32,
    pub swizzle_config: SwizzleConfig,
}

impl TileConfig for SharedTileConfig {
    fn plane_dim(&self) -> u32 {
        self.plane_dim
    }

    fn elements_in_tile_m(&self) -> u32 {
        self.tile_size.m()
    }

    fn elements_in_tile_n(&self) -> u32 {
        self.tile_size.n()
    }

    fn elements_in_tile_k(&self) -> u32 {
        self.tile_size.k()
    }

    fn swizzle_mode(&self, ident: StageIdent) -> SwizzleMode {
        match ident {
            StageIdent::Lhs => self.swizzle_config.lhs,
            StageIdent::Rhs => self.swizzle_config.rhs,
            StageIdent::Acc => self.swizzle_config.acc,
            StageIdent::Out => self.swizzle_config.out,
        }
    }
}

impl SharedTileConfig {
    pub fn new(tile_size: TileSize, plane_dim: u32, swizzle_config: SwizzleConfig) -> Self {
        SharedTileConfig {
            tile_size,
            plane_dim,
            swizzle_config,
        }
    }
}
