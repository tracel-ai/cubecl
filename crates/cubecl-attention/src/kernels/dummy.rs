use cubecl_matmul::components::{global::PartitionedStageFamily, stage::StridedStageFamily};

use crate::components::fragment::accelerated::AcceleratedFragmentAttention;
use crate::components::fragment::dummy_register::DummyRegisterFragmentAttention;
use crate::components::stage::plane::PlaneKVReuseStageAttentionFamily;
use crate::{
    components::{
        AvailableLineSizes, batch::simple::SimpleBatchAttentionFamily,
        global::simple::SimpleGlobalAttentionFamily,
    },
    kernels::Algorithm,
};

pub struct DummyRegisterAlgorithm {}
pub struct DummyAcceleratedAlgorithm {}

impl Algorithm for DummyRegisterAlgorithm {
    type FragmentAttention = DummyRegisterFragmentAttention;
    type StageAttention = PlaneKVReuseStageAttentionFamily<
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
    type FragmentAttention = AcceleratedFragmentAttention;
    type StageAttention = PlaneKVReuseStageAttentionFamily<
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
