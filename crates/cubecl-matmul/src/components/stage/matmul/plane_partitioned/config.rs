use crate::components::{
    MatrixLayout, StageIdent, TilingScheme,
    error::MatmulSetupError,
    global::{PlaneRoleConfig, RoleRuleConfig},
    stage::{NumStages, PartitionBuffering, PartitionSchedulerScheme, StageConfig},
    tile::TileConfig,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for the plane partitioned stage matmul
pub struct PlanePartitionedStageConfig<T: TileConfig> {
    pub tile_config: T,
    pub tiling_scheme: TilingScheme,
    pub quantized: bool,
    pub partition_buffering: PartitionBuffering,
    pub num_stages: NumStages,
    plane_role_config: PlaneRoleConfig,
    ordered: bool,
}

impl<T: TileConfig> StageConfig for PlanePartitionedStageConfig<T> {
    type TileConfig = T;

    fn tile_config(self) -> Self::TileConfig {
        self.tile_config
    }

    fn stage_line_size(&self, ident: StageIdent) -> u32 {
        self.tile_config.stage_line_size(ident)
    }

    fn global_line_size(&self, ident: StageIdent) -> u32 {
        self.tile_config.global_line_size(ident)
    }

    fn matrix_layout(&self, ident: StageIdent) -> MatrixLayout {
        self.tile_config.matrix_layout(ident)
    }

    fn plane_dim(&self) -> u32 {
        self.tile_config.plane_dim()
    }

    fn partition_buffering(&self) -> PartitionBuffering {
        self.partition_buffering
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

    fn quantized(&self) -> bool {
        self.quantized
    }

    fn must_sync_plane_after_execution(&self) -> bool {
        let execution_is_sync = {
            #[cfg(target_os = "macos")]
            {
                false
            }
            #[cfg(not(target_os = "macos"))]
            {
                true
            }
        };
        !execution_is_sync && self.ordered
    }

    fn partition_schedule_scheme(&self) -> PartitionSchedulerScheme {
        PartitionSchedulerScheme::Naive
    }

    fn num_stages(&self, ident: StageIdent) -> u32 {
        match ident {
            StageIdent::Lhs => self.num_stages.lhs,
            StageIdent::Rhs => self.num_stages.rhs,
            StageIdent::Acc => 1,
            StageIdent::Out => 1,
        }
    }
}

impl<T: TileConfig> PlanePartitionedStageConfig<T> {
    #[allow(clippy::too_many_arguments)]
    /// Create a new config for plane partitioned stage matmul
    ///     
    /// May return an error if:
    /// - the number of computing planes is different from the number of partitions
    /// - double buffering is enabled but there is only one tile in n
    /// - the required shared memory exceeds the available limit
    pub fn new(
        tile_config: T,
        tiling_scheme: TilingScheme,
        quantized: bool,
        partition_buffering: PartitionBuffering,
        num_stages: NumStages,
        plane_role_config: PlaneRoleConfig,
        lhs_s_size: u32,
        rhs_s_size: u32,
        eo_size: u32,
        smem_limit: u32,
        ordered: bool,
    ) -> Result<Self, MatmulSetupError> {
        Self {
            tile_config,
            tiling_scheme,
            quantized,
            partition_buffering,
            num_stages,
            plane_role_config,
            ordered,
        }
        .validate(lhs_s_size, rhs_s_size, eo_size, smem_limit)
    }

    fn validate(
        self,
        lhs_s_size: u32,
        rhs_s_size: u32,
        eo_size: u32,
        smem_limit: u32,
    ) -> Result<Self, MatmulSetupError> {
        let tiling_scheme = <Self as StageConfig>::tiling_scheme(&self);
        let num_compute_planes = <Self as StageConfig>::num_main_flow_planes(&self);
        let num_planes_needed = tiling_scheme.stage_partitions_in_stage_mn();

        if num_compute_planes != num_planes_needed {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Error: Number of compute planes {num_compute_planes} should be {num_planes_needed}."
            ))));
        }

        // TODO we should allow buffering on m dimension
        if self.partition_buffering() == PartitionBuffering::Double
            && tiling_scheme.tiles_in_stage_partition_n() < 2
        {
            return Err(MatmulSetupError::InvalidConfig(Box::new(
                "Error: Tried doing double buffering with only one tile to compute.".to_string(),
            )));
        }

        let lhs_smem_size = self.tiling_scheme.elements_in_stage_mk() * self.num_stages.lhs;
        let rhs_smem_size = self.tiling_scheme.elements_in_stage_nk() * self.num_stages.rhs;
        let out_smem_size = self.tiling_scheme.elements_in_tile_mn() * num_compute_planes;
        let smem_total_size =
            lhs_s_size * lhs_smem_size + rhs_s_size * rhs_smem_size + eo_size * out_smem_size;

        if smem_total_size > smem_limit {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "This algorithm needs {smem_total_size:?} shared memory bytes but hardware limit is {smem_limit:?}. "
            ))));
        }

        Ok(self)
    }
}
