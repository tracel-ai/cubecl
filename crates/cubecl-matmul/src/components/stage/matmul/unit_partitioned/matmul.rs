use crate::components::MatmulPrecision;
use crate::components::global::RoleRule;
use crate::components::global::UnitWriter;
use crate::components::stage::StageConfig;
use crate::components::stage::matmul::partitioned_matmul::PartitionedStageMatmul;
use crate::components::stage::matmul::partitioned_matmul::StagePartitioner;
use crate::components::stage::matmul::unit_partitioned::UnitPartitionedStageConfig;
use crate::components::tile::TileMatmul;
use crate::components::{InputPrecision, global::memory::GlobalMemoryConfig};
use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::tensor::{View, layout::Coords2d};

#[allow(type_alias_bounds)]
/// [PartitionedStageMatmul] partitioned across units
pub type UnitMatmul<
    MP: MatmulPrecision,
    TM: TileMatmul<
            <MP::Lhs as InputPrecision>::Register,
            <MP::Rhs as InputPrecision>::Register,
            <MP::Acc as InputPrecision>::Register,
        >,
    StageLhs,
    StageRhs,
    StageAcc,
    StageOut,
> = PartitionedStageMatmul<
    MP,
    TM,
    StageLhs,
    StageRhs,
    StageAcc,
    StageOut,
    UnitPartitioner,
    UnitPartitionedStageConfig<TM::Config>,
>;

/// Defines how to partition across units
pub struct UnitPartitioner {}

#[cube]
impl StagePartitioner for UnitPartitioner {
    type Writer<IP: InputPrecision> = UnitWriter<IP>;

    fn init_writer<IP: InputPrecision, S: StageConfig>(
        tensor: View<Line<IP::Global>, Coords2d, ReadWrite>,
        #[comptime] config: GlobalMemoryConfig,
        #[comptime] stage_config: S,
    ) -> Self::Writer<IP> {
        UnitWriter::<IP>::new::<S>(tensor, config, stage_config)
    }

    fn coordinates<S: StageConfig>(#[comptime] config: S) -> Coords2d {
        let plane_id = RoleRule::new(config.role_rule_config()).compute_index();

        let absolute_index = UNIT_POS_X + config.plane_dim() * plane_id;

        let num_partitions_n = config.tiling_scheme().stage_partitions_in_stage_n();
        (
            absolute_index / num_partitions_n,
            absolute_index % num_partitions_n,
        )
    }
}
