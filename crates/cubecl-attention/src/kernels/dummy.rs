use cubecl_matmul::components::{global::PartitionedStageFamily, stage::StridedStageFamily};

use crate::components::fragment::accelerated::AcceleratedAttentionMatmul;
use crate::components::fragment::dummy_register::DummyRegisterAttentionMatmul;
use crate::{
    components::{
        AvailableLineSizes, batch::simple::SimpleBatchAttentionFamily,
        global::simple::SimpleGlobalAttentionFamily,
        stage::simple_kv_reuse::SimpleKVReuseStageAttentionFamily,
    },
    kernels::Algorithm,
};

pub struct DummyRegisterAlgorithm {}
pub struct DummyAcceleratedAlgorithm {}

impl Algorithm for DummyRegisterAlgorithm {
    type FragmentAttention = DummyRegisterAttentionMatmul;
    type StageAttention = SimpleKVReuseStageAttentionFamily<
        Self::FragmentAttention,
        StridedStageFamily,
        StridedStageFamily,
        PartitionedStageFamily,
    >;
    type GlobalAttention = SimpleGlobalAttentionFamily<Self::StageAttention>;
    type BatchAttention = SimpleBatchAttentionFamily<Self::GlobalAttention>;

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
    type FragmentAttention = AcceleratedAttentionMatmul;
    type StageAttention = SimpleKVReuseStageAttentionFamily<
        Self::FragmentAttention,
        StridedStageFamily,
        StridedStageFamily,
        PartitionedStageFamily,
    >;
    type GlobalAttention = SimpleGlobalAttentionFamily<Self::StageAttention>;
    type BatchAttention = SimpleBatchAttentionFamily<Self::GlobalAttention>;

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
