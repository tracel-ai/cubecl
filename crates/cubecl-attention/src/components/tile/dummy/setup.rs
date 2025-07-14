use crate::components::{
    AttentionPrecision,
    tile::{
        TileAttentionFamily,
        dummy::{DummyTileAttention, config::DummyTileConfig},
    },
};

pub struct DummyTileAttentionFamily {}
impl TileAttentionFamily for DummyTileAttentionFamily {
    type Attention<AP: AttentionPrecision> = DummyTileAttention<AP>;

    type Config = DummyTileConfig;

    fn setup<AP: crate::components::AttentionPrecision, R: cubecl_core::Runtime>(
        client: &cubecl_core::prelude::ComputeClient<R::Server, R::Channel>,
        _problem: &crate::components::AttentionProblem,
        _selection: &crate::components::AttentionSelection,
        _line_sizes: &crate::components::AttentionLineSizes,
    ) -> Result<Self::Config, crate::components::AttentionSetupError> {
        DummyTileConfig::new(client.properties().hardware.plane_size_max)
    }
}
