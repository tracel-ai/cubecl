use crate::components::{
    Ident, InputIdent, MatmulConfig, MatrixLayout, TilingScheme,
    global::{PlaneRoleConfig, RoleRuleConfig},
    stage::{PartitionBuffering, StageConfig},
    tile::TileConfig,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for the single buffer matmul
pub struct PartitionedStageConfig<T: TileConfig> {
    pub tile_config: T,
    pub tiling_scheme: TilingScheme,
    pub quantized: bool,
    pub partition_buffering: PartitionBuffering,
    pub num_stages: NumStages,
    plane_role_config: PlaneRoleConfig,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct StageVectorization {
    /// A line size of zero means use the same vectorization as global memory.
    pub stage_line_size: u8,
    /// Still unsupported.
    pub stage_elem_padding: u8,
}

impl<T: TileConfig> StageConfig for PartitionedStageConfig<T> {
    type TileConfig = T;

    fn tile_config(self) -> Self::TileConfig {
        self.tile_config
    }

    fn stage_line_size(&self, ident: Ident) -> u32 {
        self.tile_config.stage_line_size(ident)
    }

    fn matrix_layout(&self, ident: Ident) -> MatrixLayout {
        self.tile_config.matrix_layout(ident)
    }

    fn plane_dim(&self) -> u32 {
        self.tile_config.plane_dim()
    }

    fn partition_buffering(&self) -> PartitionBuffering {
        self.partition_buffering
    }

    fn num_stages(&self, ident: InputIdent) -> u32 {
        match ident {
            InputIdent::Lhs => self.num_stages.lhs,
            InputIdent::Rhs => self.num_stages.rhs,
        }
    }

    fn tiling_scheme(&self) -> TilingScheme {
        self.tiling_scheme
    }

    fn num_main_flow_planes(&self) -> u32 {
        self.plane_role_config.main_flow_count()
    }

    fn plane_role_config(&self) -> PlaneRoleConfig {
        self.plane_role_config
    }

    fn role_rule_config(&self) -> RoleRuleConfig {
        self.plane_role_config.rule
    }
}

impl<T: TileConfig> MatmulConfig for PartitionedStageConfig<T> {}

impl<T: TileConfig> PartitionedStageConfig<T> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tile_config: T,
        tiling_scheme: TilingScheme,
        quantized: bool,
        partition_buffering: PartitionBuffering,
        num_stages: NumStages,
        plane_role_config: PlaneRoleConfig,
    ) -> Self {
        Self {
            tile_config,
            tiling_scheme,
            quantized,
            partition_buffering,
            num_stages,
            plane_role_config,
        }
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct NumStages {
    lhs: u32,
    rhs: u32,
}

impl From<(u32, u32)> for NumStages {
    fn from(value: (u32, u32)) -> Self {
        NumStages {
            lhs: value.0,
            rhs: value.1,
        }
    }
}
