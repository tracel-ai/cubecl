use crate::components::InputPrecision;
use crate::components::MatmulPrecision;
use crate::components::global::PlaneWriter;
use crate::components::global::RoleRule;
use crate::components::stage::StageConfig;
use crate::components::stage::matmul::partitioned_matmul::PartitionedStageMatmul;
use crate::components::stage::matmul::partitioned_matmul::StagePartitioner;
use crate::components::stage::matmul::plane_partitioned::PlanePartitionedStageConfig;
use crate::components::tile::TileMatmul;
use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::tensor::{View, layout::Coords3d};

#[allow(type_alias_bounds)]
/// [PartitionedStageMatmul] partitioned across units
pub type PlaneMatmul<
    MP: MatmulPrecision,
    TMM: TileMatmul<
            <MP::Lhs as InputPrecision>::Register,
            <MP::Rhs as InputPrecision>::Register,
            <MP as MatmulPrecision>::EA,
        >,
    RL,
    RR,
> = PartitionedStageMatmul<
    MP,
    TMM,
    RL,
    RR,
    PlanePartitioner,
    PlanePartitionedStageConfig<TMM::Config>,
>;

/// Defines how to partition across planes
pub struct PlanePartitioner {}

#[cube]
impl StagePartitioner for PlanePartitioner {
    type Writer<EO: Numeric> = PlaneWriter<EO>;
    type WriteCoords = Coords3d;

    fn init_writer<EO: Numeric>(
        tensor: View<Line<EO>, Self::WriteCoords, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
    ) -> Self::Writer<EO> {
        PlaneWriter::<EO>::new(tensor, x_offset, y_offset, batch_offset)
    }

    fn coordinates<S: StageConfig>(#[comptime] config: S) -> (u32, u32) {
        let absolute_index = RoleRule::new(config.role_rule_config()).compute_index();
        let num_partitions_n = config.tiling_scheme().stage_partitions_in_stage_n();
        (
            absolute_index / num_partitions_n,
            absolute_index % num_partitions_n,
        )
    }

    fn num_primitives<S: StageConfig>(#[comptime] config: S) -> comptime_type!(u32) {
        config.num_main_flow_planes()
    }
}
