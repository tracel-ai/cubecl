use crate::components::{AttentionTilingScheme, batch::HypercubeSelection};

#[derive(Debug, Clone)]
pub struct AttentionSelection {
    pub hypercube_selection: HypercubeSelection,

    pub tiling_scheme: AttentionTilingScheme,
    pub plane_dim: u32,

    pub reuse_key_value: bool,
    pub two_rows_in_array_tile: bool,
}
