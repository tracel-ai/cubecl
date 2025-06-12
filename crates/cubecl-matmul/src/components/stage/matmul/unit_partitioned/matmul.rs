use crate::components::global::RoleRule;
use crate::components::global::UnitWriter;
use crate::components::stage::matmul::partitioned_matmul::PartitionedStageMatmul;
use crate::components::stage::matmul::partitioned_matmul::StagePartitioner;
use crate::components::stage::StageConfig;
use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};

pub type UnitMatmul<MP, TMM, RL, RR> = PartitionedStageMatmul<MP, TMM, RL, RR, UnitPartitioner>;

pub struct UnitPartitioner {}

#[cube]
impl StagePartitioner for UnitPartitioner {
    type Writer<EO: Numeric> = UnitWriter<EO>;

    fn init_writer<EO: Numeric>(
        tensor: VirtualTensor<EO, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
    ) -> Self::Writer<EO> {
        UnitWriter::<EO>::new(tensor, x_offset, y_offset, batch_offset)
    }

    fn position<S: StageConfig>(#[comptime] config: S) -> u32 {
        let plane_id = RoleRule::new(config.role_rule_config()).compute_index();

        UNIT_POS_X + config.plane_dim() * plane_id
    }

    fn num_primitives<S: StageConfig>(#[comptime] config: S) -> comptime_type!(u32) {
        config.num_main_flow_planes() * config.plane_dim()
    }
}
