use crate::{
    components::{
        batch::dummy::DummyBatchAttentionFamily, global::dummy::DummyGlobalAttentionFamily,
        stage::dummy::DummyStageAttentionFamily, tile::dummy::DummyTileAttentionFamily,
    },
    kernels::Algorithm,
};

pub struct DummyAlgorithm {}

impl Algorithm for DummyAlgorithm {
    type TileAttention = DummyTileAttentionFamily;
    type StageAttention = DummyStageAttentionFamily<Self::TileAttention>;
    type GlobalAttention = DummyGlobalAttentionFamily<Self::StageAttention>;
    type BatchAttention = DummyBatchAttentionFamily<Self::GlobalAttention>;
}
