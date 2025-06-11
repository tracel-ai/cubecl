use crate::{
    components::{
        Ident, InputIdent, MatmulConfig, MatrixLayout,
        global::{
            GlobalConfig, LoadingSides, PlaneRoleConfig, SpecializedLoadingSides, load::LoaderMode,
            multi_stage::EventLoadingMode,
        },
        stage::{self},
    },
    kernels::matmul::LoadingPrecomputeStrategy,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for the pipelined global matmul
pub struct OrderedDoubleBufferingGlobalConfig<S: stage::StageConfig> {
    pub stage_config: S,
    pub check_m_bounds: bool,
    pub check_n_bounds: bool,
    pub check_k_bounds: bool,
    pub lhs_layout: MatrixLayout,
    pub rhs_layout: MatrixLayout,
    pub lhs_line_size: u32,
    pub rhs_line_size: u32,
    pub out_line_size: u32,
    pub num_planes: u32,
    precompute_job: LoadingPrecomputeStrategy,
    loader_mode: LoaderMode,
}

impl<S: stage::StageConfig> GlobalConfig for OrderedDoubleBufferingGlobalConfig<S> {
    type StageConfig = S;

    fn stage_config(&self) -> Self::StageConfig {
        self.stage_config
    }

    fn global_line_size<I: Into<Ident>>(&self, ident: I) -> u32 {
        match ident.into() {
            Ident::Lhs => self.lhs_line_size,
            Ident::Rhs => self.rhs_line_size,
            Ident::Out => self.out_line_size,
        }
    }

    fn matrix_layout<I: Into<Ident>>(&self, ident: I) -> MatrixLayout {
        match ident.into() {
            Ident::Lhs => self.lhs_layout,
            Ident::Rhs => self.rhs_layout,
            Ident::Out => self.stage_config.matrix_layout(Ident::Out),
        }
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
}

impl<S: stage::StageConfig> MatmulConfig for OrderedDoubleBufferingGlobalConfig<S> {}

impl<S: stage::StageConfig> OrderedDoubleBufferingGlobalConfig<S> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        stage_config: S,
        check_m_bounds: bool,
        check_n_bounds: bool,
        check_k_bounds: bool,
        lhs_layout: MatrixLayout,
        rhs_layout: MatrixLayout,
        lhs_line_size: u32,
        rhs_line_size: u32,
        out_line_size: u32,
        num_planes: u32,
        precompute_job: LoadingPrecomputeStrategy,
        loader_mode: LoaderMode,
    ) -> Self {
        Self {
            stage_config,
            check_m_bounds,
            check_n_bounds,
            check_k_bounds,
            lhs_layout,
            rhs_layout,
            lhs_line_size,
            rhs_line_size,
            out_line_size,
            num_planes,
            precompute_job,
            loader_mode,
        }
    }
}
