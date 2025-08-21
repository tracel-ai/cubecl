use crate::components::{batch::HypercubeSelection, tile::dummy::AttentionTileSize};

#[derive(Debug, Clone)]
pub struct AttentionSelection {
    pub hypercube_selection: HypercubeSelection,

    pub attention_tile_size: AttentionTileSize,
    pub plane_dim: u32,
}
