use cubecl_core::{CubeDim, Runtime, client::ComputeClient};

use crate::components::{
    LoadingPrecomputeStrategy, MatmulIdent, MatrixLayout, TilingScheme,
    error::MatmulSetupError,
    global::{
        self, GlobalConfig, GlobalReaderConfig, LoadingSides, PlaneRoleConfig, RoleRuleConfig,
        SpecializedLoadingSides,
        multi_stage::EventLoadingMode,
        read::{LoadingValidation, ReaderMode},
        shared::shared_global_config_validation,
    },
    stage::{self, StageConfig, StageMemoryConfig},
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for simple matmul
pub struct SimpleConfig<S: stage::StageConfig> {
    stage_config: S,
    num_planes: u32,
    check_m_bounds: bool,
    check_n_bounds: bool,
    check_k_bounds: bool,
    pub k_step: u32,
    precompute_job: LoadingPrecomputeStrategy,
    reader_mode: ReaderMode,
}

impl<S: StageConfig> GlobalConfig for SimpleConfig<S> {
    type StageConfig = S;
    type LhsReaderConfig = Self;
    type RhsReaderConfig = Self;

    fn lhs_reader_config(&self) -> Self::LhsReaderConfig {
        *self
    }

    fn rhs_reader_config(&self) -> Self::RhsReaderConfig {
        *self
    }

    fn stage_config(&self) -> Self::StageConfig {
        self.stage_config
    }

    fn global_line_size(&self, ident: MatmulIdent) -> u32 {
        self.stage_config.global_line_size(ident.into_stage())
    }

    fn matrix_layout(&self, ident: MatmulIdent) -> MatrixLayout {
        self.stage_config.matrix_layout(ident.into_stage())
    }

    fn plane_dim(&self) -> u32 {
        self.stage_config.plane_dim()
    }

    fn check_row_bounds(&self, ident: MatmulIdent) -> bool {
        match ident {
            MatmulIdent::Lhs => self.check_m_bounds,
            MatmulIdent::Rhs => self.check_k_bounds,
            MatmulIdent::Out => self.check_m_bounds,
        }
    }

    fn check_col_bounds(&self, ident: MatmulIdent) -> bool {
        match ident {
            MatmulIdent::Lhs => self.check_k_bounds,
            MatmulIdent::Rhs => self.check_n_bounds,
            MatmulIdent::Out => self.check_n_bounds,
        }
    }

    fn check_k_bounds(&self) -> bool {
        self.check_k_bounds
    }

    fn cube_dim(&self) -> CubeDim {
        CubeDim::new_2d(
            <Self as global::GlobalConfig>::plane_dim(self),
            self.num_planes,
        )
    }

    fn num_stages(&self, _ident: MatmulIdent) -> u32 {
        1
    }

    fn role_rule_config(&self) -> RoleRuleConfig {
        self.plane_role_config().rule
    }
}

impl<S: stage::StageConfig> GlobalReaderConfig for SimpleConfig<S> {
    fn stage_memory_config(&self, ident: MatmulIdent) -> StageMemoryConfig {
        self.stage_config().stage_memory_config(ident.into_stage())
    }

    fn tiling_scheme(&self) -> TilingScheme {
        self.stage_config().tiling_scheme()
    }

    fn global_line_size(&self, ident: MatmulIdent) -> u32 {
        <Self as GlobalConfig>::global_line_size(self, ident)
    }

    fn matrix_layout(&self, ident: MatmulIdent) -> MatrixLayout {
        <Self as GlobalConfig>::matrix_layout(self, ident)
    }

    fn num_loading_planes(&self, _ident: MatmulIdent) -> u32 {
        // Specialized is not available
        <Self as GlobalConfig>::stage_config(self).num_main_flow_planes()
    }

    fn plane_role_config(&self) -> PlaneRoleConfig {
        self.stage_config.plane_role_config()
    }

    fn specialized_loading_sides(&self) -> SpecializedLoadingSides {
        SpecializedLoadingSides {
            main_flow: LoadingSides::Both,
            // Specialized is not available
            load_only: LoadingSides::None,
        }
    }

    fn plane_dim(&self) -> u32 {
        <Self as GlobalConfig>::plane_dim(self)
    }

    fn check_row_bounds(&self, ident: MatmulIdent) -> bool {
        <Self as GlobalConfig>::check_row_bounds(self, ident)
    }

    fn check_col_bounds(&self, ident: MatmulIdent) -> bool {
        <Self as GlobalConfig>::check_col_bounds(self, ident)
    }

    fn precompute_job(&self) -> bool {
        self.precompute_job.into()
    }

    fn reader_mode(&self) -> ReaderMode {
        self.reader_mode
    }

    fn num_stages(&self, _ident: MatmulIdent) -> u32 {
        1
    }

    fn event_loading_mode(&self, _ident: MatmulIdent) -> EventLoadingMode {
        // Should not be used for simple
        unreachable!()
    }
}

impl<S: stage::StageConfig> SimpleConfig<S> {
    #[allow(clippy::too_many_arguments)]
    /// Create a new config for simple global matmul
    ///
    /// May return an error if:
    /// - a reader is invalid
    /// - CubeDim is too big
    /// - Barriers are not available
    pub fn new<LL: LoadingValidation, RL: LoadingValidation, R: Runtime>(
        client: &ComputeClient<R::Server>,
        stage_config: S,
        num_planes: u32,
        check_m_bounds: bool,
        check_n_bounds: bool,
        check_k_bounds: bool,
        k_step: u32,
        precompute_job: LoadingPrecomputeStrategy,
        reader_mode: ReaderMode,
    ) -> Result<Self, MatmulSetupError> {
        Self {
            stage_config,
            num_planes,
            check_m_bounds,
            check_n_bounds,
            check_k_bounds,
            k_step,
            precompute_job,
            reader_mode,
        }
        .validate::<LL, RL, R>(client)
    }

    fn validate<LL: LoadingValidation, RL: LoadingValidation, R: Runtime>(
        self,
        client: &ComputeClient<R::Server>,
    ) -> Result<Self, MatmulSetupError> {
        LL::check::<Self, R>(client, &self, MatmulIdent::Lhs)?;
        RL::check::<Self, R>(client, &self, MatmulIdent::Rhs)?;
        shared_global_config_validation(self)?;

        Ok(self)
    }
}
