use crate::components::{stage::matmul::partition::SharedPartitionMatmulConfig, tile::TileConfig};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for the unit partitioned stage matmul
pub struct UnitPartitionedStageConfig<TC: TileConfig> {
    pub shared: SharedPartitionMatmulConfig<TC>,
}

impl<TC: TileConfig> UnitPartitionedStageConfig<TC> {
    pub fn from_shared_partition_config(shared: SharedPartitionMatmulConfig<TC>) -> Self {
        Self { shared }
    }
}

// impl<T: TileConfig> StageConfig for UnitPartitionedStageConfig<T> {
//     type TileConfig = T;

//     fn elements_in_stage_m(&self) -> u32 {
//         self.
//     }

//     fn elements_in_stage_n(&self) -> u32 {
//         todo!()
//     }

//     fn elements_in_stage_k(&self) -> u32 {
//         todo!()
//     }
// }

// pub tile_config: T,
// pub tiling_scheme: TilingScheme,
// pub quantized: bool,
// pub partition_buffering: PartitionBuffering,
// pub num_stages: NumStages,
// plane_role_config: PlaneRoleConfig,
// // rm, needed only for compute config
// ordered: bool,

// fn tile_config(self) -> Self::TileConfig {
//     self.tile_config
// }

// // delegate to stage memory config
// fn stage_line_size(&self, ident: StageIdent) -> u32 {
//     self.tile_config.stage_line_size(ident)
// }

// // rm
// fn global_line_size(&self, ident: StageIdent) -> u32 {
//     self.tile_config.global_line_size(ident)
// }

// // ensure it means the layout of the stage
// // + delegate to stage memory config
// fn matrix_layout(&self, ident: StageIdent) -> MatrixLayout {
//     self.tile_config.matrix_layout(ident)
// }

// // rm should be part of a compute config
// // then also rm from tile_config
// fn plane_dim(&self) -> u32 {
//     self.tile_config.plane_dim()
// }

// fn partition_buffering(&self) -> PartitionBuffering {
//     self.partition_buffering
// }

// // rm
// fn tiling_scheme(&self) -> TilingScheme {
//     self.tiling_scheme
// }

// // delegate to plane_role_config better
// fn num_main_flow_planes(&self) -> u32 {
//     self.plane_role_config.main_flow_count()
// }

// fn plane_role_config(&self) -> PlaneRoleConfig {
//     self.plane_role_config
// }

// // delegate to plane_role_config better
// fn role_rule_config(&self) -> RoleRuleConfig {
//     self.plane_role_config.rule
// }

// // rm
// fn quantized(&self) -> bool {
//     self.quantized
// }

// // rm part of compute config
// fn must_sync_plane_after_execution(&self) -> bool {
//     let execution_is_sync = {
//         #[cfg(target_os = "macos")]
//         {
//             false
//         }
//         #[cfg(not(target_os = "macos"))]
//         {
//             true
//         }
//     };
//     !execution_is_sync && self.ordered
// }

// // keep
// fn partition_schedule_scheme(&self) -> PartitionSchedulerScheme {
//     PartitionSchedulerScheme::Naive
// }

// // rm doesnt event seems used
// fn num_stages(&self, ident: StageIdent) -> u32 {
//     match ident {
//         StageIdent::Lhs => self.num_stages.lhs,
//         StageIdent::Rhs => self.num_stages.rhs,
//         StageIdent::Acc => 1,
//         StageIdent::Out => 1,
//     }
// }
