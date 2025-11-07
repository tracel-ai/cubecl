use cubecl_matmul::components::{global::PartitionedStageFamily, stage::StridedStageFamily};

use crate::components::stage::plane::PlanePartitionStageAttentionFamily;
use crate::components::tile::accelerated::BlackboxAcceleratedFragmentAttention;
use crate::{
    components::{
        AvailableLineSizes, batch::simple::SimpleBatchAttentionFamily,
        global::simple::SimpleGlobalAttentionFamily,
    },
    kernels::Algorithm,
};

pub struct BlackboxAcceleratedAlgorithm {}

impl Algorithm for BlackboxAcceleratedAlgorithm {
    type FragmentAttention = BlackboxAcceleratedFragmentAttention;
    type StageAttention = PlanePartitionStageAttentionFamily<
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
