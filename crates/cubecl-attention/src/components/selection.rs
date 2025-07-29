use cubecl_matmul::components::TileSize;

use crate::components::batch::HypercubeSelection;

#[derive(Debug, Clone)]
pub struct AttentionSelection {
    pub hypercube_selection: HypercubeSelection,

    pub value_tile_size: TileSize,
    pub score_tile_size: TileSize,
    pub plane_dim: u32,
}
