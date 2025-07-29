use cubecl_matmul::components::{stage::FullReaderFamily, tile::accelerated::AcceleratedMatmul};

use crate::{
    components::{
        AvailableLineSizes, batch::dummy::DummyBatchAttentionFamily,
        global::dummy::DummyGlobalAttentionFamily, stage::dummy::DummyStageAttentionFamily,
    },
    kernels::Algorithm,
};

pub struct DummyAlgorithm {}

impl Algorithm for DummyAlgorithm {
    type ScoreTileMatmul = AcceleratedMatmul;
    type ValueTileMatmul = AcceleratedMatmul;
    type StageAttention =
        DummyStageAttentionFamily<Self::ScoreTileMatmul, Self::ValueTileMatmul, FullReaderFamily>;
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
