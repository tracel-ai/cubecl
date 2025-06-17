use cubecl_core::{CubeDim, Runtime, client::ComputeClient};

use crate::{
    components::{
        Ident, InputIdent, MatmulConfig, MatmulPrecision, MatrixLayout,
        global::{
            GlobalConfig, LoadingSides, PlaneRoleConfig, SpecializedLoadingSides,
            load::{LoaderMode, LoadingValidation},
            multi_stage::EventLoadingMode,
            shared::shared_global_config_validation,
        },
        stage::{self},
    },
    kernels::{MatmulSetupError, matmul::LoadingPrecomputeStrategy},
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for the pipelined global matmul
pub struct OrderedDoubleBufferingGlobalConfig<S: stage::StageConfig> {
    pub stage_config: S,
    num_planes: u32,
    pub check_m_bounds: bool,
    pub check_n_bounds: bool,
    pub check_k_bounds: bool,
    precompute_job: LoadingPrecomputeStrategy,
    loader_mode: LoaderMode,
}

impl<S: stage::StageConfig> GlobalConfig for OrderedDoubleBufferingGlobalConfig<S> {
    type StageConfig = S;

    fn stage_config(&self) -> Self::StageConfig {
        self.stage_config
    }

    fn global_line_size<I: Into<Ident>>(&self, ident: I) -> u32 {
        self.stage_config.global_line_size(ident)
    }

    fn matrix_layout<I: Into<Ident>>(&self, ident: I) -> MatrixLayout {
        self.stage_config.matrix_layout(ident)
    }

    fn plane_dim(&self) -> u32 {
        self.stage_config.plane_dim()
    }

    fn check_row_bounds<I: Into<Ident>>(&self, ident: I) -> bool {
        match ident.into() {
            Ident::Lhs => self.check_m_bounds,
            Ident::Rhs => self.check_k_bounds,
            Ident::Out => self.check_m_bounds,
        }
    }

    fn check_col_bounds<I: Into<Ident>>(&self, ident: I) -> bool {
        match ident.into() {
            Ident::Lhs => self.check_k_bounds,
            Ident::Rhs => self.check_n_bounds,
            Ident::Out => self.check_n_bounds,
        }
    }

    fn check_k_bounds(&self) -> bool {
        self.check_k_bounds
    }

    fn precompute_job(&self) -> bool {
        self.precompute_job.into()
    }

    fn num_stages(&self, ident: InputIdent) -> u32 {
        match ident {
            InputIdent::Lhs => 1,
            InputIdent::Rhs => 2,
        }
    }

    fn loader_mode(&self) -> LoaderMode {
        self.loader_mode
    }

    fn event_loading_mode(&self, ident: InputIdent) -> EventLoadingMode {
        match ident {
            InputIdent::Lhs => EventLoadingMode::Ordered,
            InputIdent::Rhs => EventLoadingMode::Relaxed,
        }
    }

    fn plane_role_config(&self) -> PlaneRoleConfig {
        self.stage_config.plane_role_config()
    }

    fn specialized_loading_sides(&self) -> SpecializedLoadingSides {
        SpecializedLoadingSides {
            main_flow: LoadingSides::Lhs,
            load_only: LoadingSides::Rhs,
        }
    }

    fn num_loading_planes<I: Into<Ident>>(&self, ident: I) -> u32 {
        self.specialized_loading_sides().num_loading_planes(
            self.plane_role_config().has_specialization(),
            ident.into().as_input_ident(),
            self.plane_role_config().plane_roles,
        )
    }

    fn cube_dim(&self) -> CubeDim {
        CubeDim::new_2d(self.plane_dim(), self.num_planes)
    }
}

impl<S: stage::StageConfig> MatmulConfig for OrderedDoubleBufferingGlobalConfig<S> {}

impl<S: stage::StageConfig> OrderedDoubleBufferingGlobalConfig<S> {
    #[allow(clippy::too_many_arguments)]
    pub fn new<LL: LoadingValidation, RL: LoadingValidation, MP: MatmulPrecision, R: Runtime>(
        _client: &ComputeClient<R::Server, R::Channel>,
        stage_config: S,
        num_planes: u32,
        check_m_bounds: bool,
        check_n_bounds: bool,
        check_k_bounds: bool,
        precompute_job: LoadingPrecomputeStrategy,
        loader_mode: LoaderMode,
    ) -> Result<Self, MatmulSetupError> {
        Self {
            stage_config,
            num_planes,
            check_m_bounds,
            check_n_bounds,
            check_k_bounds,
            precompute_job,
            loader_mode,
        }
        .validate::<LL, RL>()
    }

    fn validate<LL: LoadingValidation, RL: LoadingValidation>(
        self,
    ) -> Result<Self, MatmulSetupError> {
        LL::check::<Self>(&self, Ident::Lhs)?;
        RL::check::<Self>(&self, Ident::Rhs)?;
        shared_global_config_validation(self)?;
        if self.tiling_scheme().stage_partitions_in_stage_n() > 1 {
            return Err(MatmulSetupError::InvalidConfig(Box::new(
                "Ordered does not support number of stage partitions > 1 in n",
            )));
        }

        Ok(self)
    }
}
