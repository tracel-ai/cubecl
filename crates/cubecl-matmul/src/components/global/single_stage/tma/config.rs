use std::any::TypeId;

use cubecl_core::{CubeDim, Feature, Runtime, TmaFeature, client::ComputeClient, tf32};

use crate::{
    components::{
        Ident, InputIdent, MatmulConfig, MatmulPrecision, MatrixLayout,
        global::{
            self, LoadingSides, PlaneRoleConfig, SpecializedLoadingSides,
            load::{LoaderMode, LoadingValidation},
            multi_stage::EventLoadingMode,
        },
        stage,
    },
    kernels::{MatmulAvailabilityError, MatmulSetupError, matmul::LoadingPrecomputeStrategy},
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for single stage matmuls
pub struct SimpleTmaConfig<S: stage::StageConfig> {
    stage_config: S,
    num_planes: u32,
    check_m_bounds: bool,
    check_n_bounds: bool,
    check_k_bounds: bool,
    pub k_step: u32,
    precompute_job: LoadingPrecomputeStrategy,
    loader_mode: LoaderMode,
}

impl<S: stage::StageConfig> global::GlobalConfig for SimpleTmaConfig<S> {
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

    fn num_stages(&self, _ident: InputIdent) -> u32 {
        1
    }

    fn precompute_job(&self) -> bool {
        self.precompute_job.into()
    }

    fn loader_mode(&self) -> LoaderMode {
        self.loader_mode
    }

    fn event_loading_mode(&self, _ident: InputIdent) -> EventLoadingMode {
        EventLoadingMode::Relaxed
    }

    fn plane_role_config(&self) -> PlaneRoleConfig {
        self.stage_config.plane_role_config()
    }

    fn num_loading_planes<I: Into<Ident>>(&self, _ident: I) -> u32 {
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

impl<S: stage::StageConfig> MatmulConfig for SimpleTmaConfig<S> {}

impl<S: stage::StageConfig> SimpleTmaConfig<S> {
    #[allow(clippy::too_many_arguments)]
    pub fn new<LL: LoadingValidation, RL: LoadingValidation, MP: MatmulPrecision, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
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
        .validate::<LL, RL>()?
        .check_availability::<MP, R>(client)
    }

    fn validate<LL: LoadingValidation, RL: LoadingValidation>(
        self,
    ) -> Result<Self, MatmulSetupError> {
        LL::check(&self, Ident::Lhs)?;
        RL::check(&self, Ident::Rhs)?;
        Ok(self)
    }

    fn check_availability<MP: MatmulPrecision, R: Runtime>(
        self,
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> Result<Self, MatmulSetupError> {
        let ei_id = TypeId::of::<MP::EI>();
        let es_id = TypeId::of::<MP::ES>();
        let is_tf32 = ei_id == TypeId::of::<f32>() && es_id == TypeId::of::<tf32>();

        if ei_id != es_id && !is_tf32 {
            return Err(MatmulSetupError::Unavailable(
                MatmulAvailabilityError::TmaUnavailable,
            ));
        }

        let ei_id = TypeId::of::<MP::EI>();
        let es_id = TypeId::of::<MP::ES>();
        let is_tf32 = ei_id == TypeId::of::<f32>() && es_id == TypeId::of::<tf32>();

        if ei_id != es_id && !is_tf32 {
            return Err(MatmulSetupError::Unavailable(
                MatmulAvailabilityError::TmaUnavailable,
            ));
        }

        if !client
            .properties()
            .feature_enabled(Feature::Tma(TmaFeature::Base))
        {
            return Err(MatmulSetupError::Unavailable(
                MatmulAvailabilityError::TmaUnavailable,
            ));
        }

        Ok(self)
    }
}
