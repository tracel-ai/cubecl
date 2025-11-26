use cubecl_matmul::components::{global::PartitionedStageFamily, stage::StridedStageFamily};

use crate::components::stage::unit::UnitPartitionStageAttentionFamily;
use crate::components::tile::unit_register::UnitRegisterTileAttention;
use crate::{
    components::{
        AvailableLineSizes, batch::simple::SimpleBatchAttentionFamily,
        global::simple::SimpleGlobalAttentionFamily,
    },
    kernels::Algorithm,
};

pub struct UnitAlgorithm {}

impl Algorithm for UnitAlgorithm {
    type TileAttention = UnitRegisterTileAttention;
    type StageAttention = UnitPartitionStageAttentionFamily<
        Self::TileAttention,
        StridedStageFamily,
        StridedStageFamily,
        PartitionedStageFamily,
    >;
    type GlobalAttention = SimpleGlobalAttentionFamily<Self::StageAttention>;
    type BatchAttention = SimpleBatchAttentionFamily<Self::GlobalAttention>;

    fn filter_line_sizes(available_line_sizes: AvailableLineSizes) -> AvailableLineSizes {
        AvailableLineSizes {
            query: available_line_sizes.query,
            key: available_line_sizes.key,
            value: available_line_sizes.value,
            mask: available_line_sizes.mask,
            out: available_line_sizes.out,
        }
    }
}
