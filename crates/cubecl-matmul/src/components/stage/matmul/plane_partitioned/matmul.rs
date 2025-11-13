use crate::components::MatmulPrecision;
use crate::components::MatrixPrecision;
use crate::components::global::RoleRule;
use crate::components::stage::StageConfig;
use crate::components::stage::matmul::partitioned_matmul::PartitionedStageMatmul;
use crate::components::stage::matmul::partitioned_matmul::StagePartitioner;
use crate::components::stage::matmul::plane_partitioned::PlanePartitionedStageConfig;
use crate::components::tile::TileMatmul;
use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::tensor::layout::Coords2d;

#[allow(type_alias_bounds)]
/// [PartitionedStageMatmul] partitioned across units
pub type PlaneMatmul<
    MP: MatmulPrecision,
    TM: TileMatmul<
            <MP::Lhs as MatrixPrecision>::Register,
            <MP::Rhs as MatrixPrecision>::Register,
            <MP::Acc as MatrixPrecision>::Register,
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
    PlanePartitioner,
    PlanePartitionedStageConfig<TM::Config>,
>;

/// Defines how to partition across planes
pub struct PlanePartitioner {}

#[cube]
impl StagePartitioner for PlanePartitioner {
    fn coordinates<S: StageConfig>(#[comptime] config: S) -> Coords2d {
        let absolute_index = RoleRule::new(config.role_rule_config()).compute_index();
        let num_partitions_n = config.tiling_scheme().stage_partitions_in_stage_n();
        (
            absolute_index / num_partitions_n,
            absolute_index % num_partitions_n,
        )
    }
}
