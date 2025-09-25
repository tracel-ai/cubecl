use crate::components::global::RoleRule;
use crate::components::stage::StageConfig;
use crate::components::stage::matmul::partitioned_matmul::PartitionedStageMatmul;
use crate::components::stage::matmul::partitioned_matmul::StagePartitioner;
use crate::components::stage::matmul::plane_partitioned::PlanePartitionedStageConfig;
use crate::components::tile::TileMatmul;
use crate::components::{InputPrecision, global::memory::GlobalMemoryConfig};
use crate::components::{MatmulPrecision, stage::StageMemoryConfig};
use crate::components::{StageIdent, global::PlaneWriter};
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
    RL,
    RR,
    RA,
> = PartitionedStageMatmul<
    MP,
    TMM,
    RL,
    RR,
    RA,
    PlanePartitioner,
    PlanePartitionedStageConfig<TMM::Config>,
>;

/// Defines how to partition across planes
pub struct PlanePartitioner {}

#[cube]
impl StagePartitioner for PlanePartitioner {
    type Writer<EO: Numeric> = PlaneWriter<EO>;

    fn init_writer<EO: Numeric>(
        tensor: View<Line<EO>, Coords2d, ReadWrite>,
        #[comptime] config: GlobalMemoryConfig,
    ) -> Self::Writer<EO> {
        PlaneWriter::<EO>::new(tensor, config)
    }

    fn coordinates<S: StageConfig>(#[comptime] config: S) -> Coords2d {
        let absolute_index = RoleRule::new(config.role_rule_config()).compute_index();
        let num_partitions_n = config.tiling_scheme().stage_partitions_in_stage_n();
        (
            absolute_index / num_partitions_n,
            absolute_index % num_partitions_n,
        )
    }

    fn stage_memory_config<S: StageConfig>(
        #[comptime] config: S,
    ) -> comptime_type!(StageMemoryConfig) {
        comptime! {
            let planes = config.num_main_flow_planes();
            let size_n = config.tiling_scheme().stage_partitions_in_stage_n();
            let base = config.stage_memory_config(StageIdent::Acc);
            StageMemoryConfig {
                tiles_in_stage_row: planes / size_n,
                tiles_in_stage_col: size_n,
                ..base
            }
        }
    }
}
