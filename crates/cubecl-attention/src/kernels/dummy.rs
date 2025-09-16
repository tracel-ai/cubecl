use cubecl_matmul::components::stage::FullStageReaderFamily;

use crate::{
    components::{
        AvailableLineSizes,
        batch::dummy::DummyBatchAttentionFamily,
        global::dummy::DummyGlobalAttentionFamily,
        stage::dummy::DummyStageAttentionFamily,
        tile::dummy::{DummyTileAttentionFamily, dummy_register::DummyRegisterFlashMatmul},
    },
    kernels::Algorithm,
};

pub struct DummyAlgorithm {}

impl Algorithm for DummyAlgorithm {
    // type TileAttention = DummyTileAttentionFamily<AcceleratedFlashMatmul>;
    type TileAttention = DummyTileAttentionFamily<DummyRegisterFlashMatmul>;
    type StageAttention = DummyStageAttentionFamily<Self::TileAttention, FullStageReaderFamily>;
    type GlobalAttention = DummyGlobalAttentionFamily<Self::StageAttention>;
    type BatchAttention = DummyBatchAttentionFamily<Self::GlobalAttention>;

    fn filter_line_sizes(_available_line_sizes: AvailableLineSizes) -> AvailableLineSizes {
        AvailableLineSizes {
            query: vec![1],
            key: vec![1],
            value: vec![1],
            mask: vec![1],
            out: vec![1],
        }
    }
}
