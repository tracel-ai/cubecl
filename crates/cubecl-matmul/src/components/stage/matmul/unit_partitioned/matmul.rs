use crate::components::global::UnitWriter;
use crate::components::stage::StageConfig;
use crate::components::stage::matmul::partitioned_matmul::PartitionedStageMatmul;
use crate::components::stage::matmul::partitioned_matmul::StagePartitioner;
use crate::components::stage::matmul::unit_partitioned::UnitPartitionedStageConfig;
use crate::components::tile::TileMatmul;
use crate::components::{InputPrecision, global::memory::GlobalMemoryConfig};
use crate::components::{MatmulPrecision, StageIdent};
use crate::components::{global::RoleRule, stage::StageMemoryConfig};
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
    RL,
    RR,
    RA,
> = PartitionedStageMatmul<
    MP,
    TM,
    RL,
    RR,
    RA,
    UnitPartitioner,
    UnitPartitionedStageConfig<TM::Config>,
>;

/// Defines how to partition across units
pub struct UnitPartitioner {}

#[cube]
impl StagePartitioner for UnitPartitioner {
    type Writer<EO: Numeric> = UnitWriter<EO>;

    fn init_writer<EO: Numeric>(
        tensor: View<Line<EO>, Coords2d, ReadWrite>,
        #[comptime] config: GlobalMemoryConfig,
    ) -> Self::Writer<EO> {
        UnitWriter::<EO>::new(tensor, config)
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

    fn stage_memory_config<S: StageConfig>(
        #[comptime] config: S,
    ) -> comptime_type!(StageMemoryConfig) {
        comptime! {
            let units = config.num_main_flow_planes() * config.plane_dim();
            let size_n = config.tiling_scheme().stage_partitions_in_stage_n();
            let base = config.stage_memory_config(StageIdent::Acc);
            StageMemoryConfig {
                tiles_in_stage_row: units / size_n,
                tiles_in_stage_col: size_n,
                ..base
            }
        }
    }
}
