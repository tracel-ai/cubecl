use cubecl_matmul::components::{global::PartitionedStageFamily, stage::StridedStageFamily};

use crate::components::fragment::accelerated::AcceleratedAttentionMatmul;
use crate::components::fragment::dummy_register::DummyRegisterAttentionMatmul;
use crate::{
    components::{
        AvailableLineSizes, batch::simple::DummyBatchAttentionFamily,
        global::simple::DummyGlobalAttentionFamily, stage::simple_kv_reuse::DummyStageAttentionFamily,
        tile::dummy::DummyTileAttentionFamily,
    },
    kernels::Algorithm,
};

pub struct DummyRegisterAlgorithm {}
pub struct DummyAcceleratedAlgorithm {}

impl Algorithm for DummyRegisterAlgorithm {
    type TileAttention = DummyTileAttentionFamily<DummyRegisterAttentionMatmul>;
    type StageAttention = DummyStageAttentionFamily<
        Self::TileAttention,
        StridedStageFamily,
        StridedStageFamily,
        PartitionedStageFamily,
    >;
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

impl Algorithm for DummyAcceleratedAlgorithm {
    type TileAttention = DummyTileAttentionFamily<AcceleratedAttentionMatmul>;
    type StageAttention = DummyStageAttentionFamily<
        Self::TileAttention,
        StridedStageFamily,
        StridedStageFamily,
        PartitionedStageFamily,
    >;
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
