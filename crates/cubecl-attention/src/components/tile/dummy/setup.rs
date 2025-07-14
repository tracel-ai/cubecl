use crate::components::{
    AttentionPrecision,
    tile::{
        TileAttentionFamily,
        dummy::{DummyTileAttention, config::DummyTileConfig},
    },
};

pub struct DummyTileAttentionFamily {}
impl TileAttentionFamily for DummyTileAttentionFamily {
    type Attention<AP: AttentionPrecision> = DummyTileAttention;

    type Config = DummyTileConfig;

    fn setup<AP: crate::components::AttentionPrecision, R: cubecl_core::Runtime>(
        client: &cubecl_core::prelude::ComputeClient<R::Server, R::Channel>,
        problem: &crate::components::AttentionProblem,
        selection: &crate::components::AttentionSelection,
        line_sizes: &crate::components::AttentionLineSizes,
    ) -> Result<Self::Config, crate::components::AttentionSetupError> {
        todo!()
    }
}
