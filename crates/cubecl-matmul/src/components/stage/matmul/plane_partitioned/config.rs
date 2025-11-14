use crate::components::{
    stage::{StageConfig, matmul::partition::SharedPartitionMatmulConfig},
    tile::TileConfig,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for the plane partitioned stage matmul
pub struct PlanePartitionedStageConfig<TC: TileConfig> {
    pub shared: SharedPartitionMatmulConfig<TC>,
}

impl<TC: TileConfig> PlanePartitionedStageConfig<TC> {
    pub fn from_shared_partition_config(shared: SharedPartitionMatmulConfig<TC>) -> Self {
        Self { shared }
    }
}

impl<T: TileConfig> StageConfig for PlanePartitionedStageConfig<T> {
    type TileConfig = T;
}

// fn tile_config(self) -> Self::TileConfig {
//     self.tile_config
// }

// fn stage_line_size(&self, ident: StageIdent) -> u32 {
//     self.tile_config.stage_line_size(ident)
// }

// fn global_line_size(&self, ident: StageIdent) -> u32 {
//     self.tile_config.global_line_size(ident)
// }

// fn matrix_layout(&self, ident: StageIdent) -> MatrixLayout {
//     self.tile_config.matrix_layout(ident)
// }

// fn plane_dim(&self) -> u32 {
//     self.tile_config.plane_dim()
// }

// fn partition_buffering(&self) -> PartitionBuffering {
//     self.partition_buffering
// }

// fn tiling_scheme(&self) -> TilingScheme {
//     self.tiling_scheme
// }

// fn num_main_flow_planes(&self) -> u32 {
//     self.plane_role_config.main_flow_count()
// }

// fn plane_role_config(&self) -> PlaneRoleConfig {
//     self.plane_role_config
// }

// fn role_rule_config(&self) -> RoleRuleConfig {
//     self.plane_role_config.rule
// }

// fn quantized(&self) -> bool {
//     self.quantized
// }

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

// fn partition_schedule_scheme(&self) -> PartitionSchedulerScheme {
//     PartitionSchedulerScheme::Naive
// }

// fn num_stages(&self, ident: StageIdent) -> u32 {
//     match ident {
//         StageIdent::Lhs => self.num_stages.lhs,
//         StageIdent::Rhs => self.num_stages.rhs,
//         StageIdent::Acc => 1,
//         StageIdent::Out => 1,
//     }
// }

// fn tile_config(self) -> Self::TileConfig {
//     self.tile_config
// }

// fn stage_line_size(&self, ident: StageIdent) -> u32 {
//     self.tile_config.stage_line_size(ident)
// }

// fn global_line_size(&self, ident: StageIdent) -> u32 {
//     self.tile_config.global_line_size(ident)
// }

// fn matrix_layout(&self, ident: StageIdent) -> MatrixLayout {
//     self.tile_config.matrix_layout(ident)
// }

// fn plane_dim(&self) -> u32 {
//     self.tile_config.plane_dim()
// }

// fn partition_buffering(&self) -> PartitionBuffering {
//     self.partition_buffering
// }

// fn tiling_scheme(&self) -> TilingScheme {
//     self.tiling_scheme
// }

// fn num_main_flow_planes(&self) -> u32 {
//     self.plane_role_config.main_flow_count()
// }

// fn plane_role_config(&self) -> PlaneRoleConfig {
//     self.plane_role_config
// }

// fn role_rule_config(&self) -> RoleRuleConfig {
//     self.plane_role_config.rule
// }

// fn quantized(&self) -> bool {
//     self.quantized
// }

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

// fn partition_schedule_scheme(&self) -> PartitionSchedulerScheme {
//     PartitionSchedulerScheme::Naive
// }

// fn num_stages(&self, ident: StageIdent) -> u32 {
//     match ident {
//         StageIdent::Lhs => self.num_stages.lhs,
//         StageIdent::Rhs => self.num_stages.rhs,
//         StageIdent::Acc => 1,
//         StageIdent::Out => 1,
//     }
// }
