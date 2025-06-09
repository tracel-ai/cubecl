use crate::{
    components::{
        Ident, InputIdent, MatmulConfig, MatrixLayout,
        global::{self, SpecializerConfig, load::LoaderMode},
        stage,
    },
    kernels::matmul::LoadingPrecomputeStrategy,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for single stage matmuls
pub struct Config<S: stage::StageConfig> {
    stage_config: S,
    check_m_bounds: bool,
    check_n_bounds: bool,
    check_k_bounds: bool,
    lhs_layout: MatrixLayout,
    rhs_layout: MatrixLayout,
    lhs_line_size: u32,
    rhs_line_size: u32,
    out_line_size: u32,
    pub k_step: u32,
    precompute_job: LoadingPrecomputeStrategy,
    loader_mode: LoaderMode,
}

impl<S: stage::StageConfig> global::GlobalConfig for Config<S> {
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

    fn num_stages(&self, _ident: InputIdent) -> u32 {
        1
    }

    fn precompute_job(&self) -> bool {
        self.precompute_job.into()
    }

    fn loader_mode(&self) -> LoaderMode {
        self.loader_mode
    }

    fn num_loading_planes(&self) -> u32 {
        self.stage_config.specializer_config().loader_count()
    }

    fn specializer_config(&self) -> SpecializerConfig {
        self.stage_config.specializer_config()
    }
}

impl<S: stage::StageConfig> MatmulConfig for Config<S> {}

impl<S: stage::StageConfig> Config<S> {
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
        k_step: u32,
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
            k_step,
            precompute_job,
            loader_mode,
        }
    }
}
