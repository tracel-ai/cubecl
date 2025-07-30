use cubecl_core::{CubeDim, Runtime, client::ComputeClient};

use crate::components::{
    LoadingPrecomputeStrategy, MatmulIdent, MatmulPrecision, MatrixLayout,
    error::MatmulSetupError,
    global::{
        self, LoadingSides, PlaneRoleConfig, SpecializedLoadingSides,
        load::{LoaderMode, LoadingValidation},
        multi_stage::EventLoadingMode,
        shared::shared_global_config_validation,
    },
    stage,
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
    loader_mode: LoaderMode,
}

impl<S: stage::StageConfig> global::GlobalConfig for SimpleConfig<S> {
    type StageConfig = S;
    type StageMemoryConfig = S::StageMemoryConfig;

    fn stage_memory_config(&self) -> Self::StageMemoryConfig {
        self.stage_config.stage_memory_config()
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

    fn num_stages(&self, _ident: MatmulIdent) -> u32 {
        1
    }

    fn precompute_job(&self) -> bool {
        self.precompute_job.into()
    }

    fn loader_mode(&self) -> LoaderMode {
        self.loader_mode
    }

    fn event_loading_mode(&self, _ident: MatmulIdent) -> EventLoadingMode {
        EventLoadingMode::Relaxed
    }

    fn plane_role_config(&self) -> PlaneRoleConfig {
        self.stage_config.plane_role_config()
    }

    fn num_loading_planes(&self, _ident: MatmulIdent) -> u32 {
        // Specialized is not available
        self.stage_config().num_main_flow_planes()
    }

    fn specialized_loading_sides(&self) -> SpecializedLoadingSides {
        SpecializedLoadingSides {
            main_flow: LoadingSides::Both,
            // Specialized is not available
            load_only: LoadingSides::None,
        }
    }

    fn cube_dim(&self) -> CubeDim {
        CubeDim::new_2d(self.plane_dim(), self.num_planes)
    }
}

impl<S: stage::StageConfig> SimpleConfig<S> {
    #[allow(clippy::too_many_arguments)]
    /// Create a new config for simple global matmul
    ///
    /// May return an error if:
    /// - a loader is invalid
    /// - CubeDim is too big
    /// - Barriers are not available
    pub fn new<LL: LoadingValidation, RL: LoadingValidation, MP: MatmulPrecision, R: Runtime>(
        _client: &ComputeClient<R::Server, R::Channel>,
        stage_config: S,
        num_planes: u32,
        check_m_bounds: bool,
        check_n_bounds: bool,
        check_k_bounds: bool,
        k_step: u32,
        precompute_job: LoadingPrecomputeStrategy,
        loader_mode: LoaderMode,
    ) -> Result<Self, MatmulSetupError> {
        Self {
            stage_config,
            num_planes,
            check_m_bounds,
            check_n_bounds,
            check_k_bounds,
            k_step,
            precompute_job,
            loader_mode,
        }
        .validate::<LL, RL>()
    }

    fn validate<LL: LoadingValidation, RL: LoadingValidation>(
        self,
    ) -> Result<Self, MatmulSetupError> {
        LL::check(&self, MatmulIdent::Lhs)?;
        RL::check(&self, MatmulIdent::Rhs)?;
        shared_global_config_validation(self)?;

        Ok(self)
    }
}
