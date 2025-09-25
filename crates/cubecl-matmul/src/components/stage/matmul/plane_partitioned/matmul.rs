use crate::components::MatmulPrecision;
use crate::components::global::PlaneWriter;
use crate::components::global::RoleRule;
use crate::components::stage::StageConfig;
use crate::components::stage::matmul::partitioned_matmul::PartitionedStageMatmul;
use crate::components::stage::matmul::partitioned_matmul::StagePartitioner;
use crate::components::stage::matmul::plane_partitioned::PlanePartitionedStageConfig;
use crate::components::tile::TileMatmul;
use crate::components::{InputPrecision, global::memory::GlobalMemoryConfig};
use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::tensor::{View, layout::Coords2d};

#[allow(type_alias_bounds)]
/// [PartitionedStageMatmul] partitioned across units
pub type PlaneMatmul<
    MP: MatmulPrecision,
    TMM: TileMatmul<
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
    TMM,
    StageLhs,
    StageRhs,
    StageAcc,
    StageOut,
    PlanePartitioner,
    PlanePartitionedStageConfig<TMM::Config>,
>;

/// Defines how to partition across planes
pub struct PlanePartitioner {}

#[cube]
impl StagePartitioner for PlanePartitioner {
    type Writer<IP: InputPrecision> = PlaneWriter<IP>;

    fn init_writer<IP: InputPrecision, S: StageConfig>(
        tensor: View<Line<IP::Global>, Coords2d, ReadWrite>,
        #[comptime] config: GlobalMemoryConfig,
        #[comptime] stage_config: S,
    ) -> Self::Writer<IP> {
        PlaneWriter::<IP>::new::<S>(tensor, config, stage_config)
    }

    fn coordinates<S: StageConfig>(#[comptime] config: S) -> Coords2d {
        let absolute_index = RoleRule::new(config.role_rule_config()).compute_index();
        let num_partitions_n = config.tiling_scheme().stage_partitions_in_stage_n();
        (
            absolute_index / num_partitions_n,
            absolute_index % num_partitions_n,
        )
    }
}
