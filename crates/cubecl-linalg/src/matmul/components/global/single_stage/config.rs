use crate::matmul::components::{
    global::{self, LoadMode},
    stage::{self, TilingLayout},
    Ident, MatmulConfig, MatrixLayout, StageTiling,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for single stage matmuls
pub struct Config<S: stage::StageConfig> {
    smm_config: S,
    check_m_bounds: bool,
    check_n_bounds: bool,
    check_k_bounds: bool,
    lhs_layout: MatrixLayout,
    rhs_layout: MatrixLayout,
    lhs_line_size: u32,
    rhs_line_size: u32,
    out_line_size: u32,
    pub k_step: u32,
    load_mode: LoadMode,
}

impl<S: stage::StageConfig> global::GlobalConfig for Config<S> {
    type SmmConfig = S;

    fn to_smm_config(&self) -> Self::SmmConfig {
        self.smm_config
    }

    fn global_line_size(&self, ident: Ident) -> u32 {
        match ident {
            Ident::Lhs => self.lhs_line_size,
            Ident::Rhs => self.rhs_line_size,
            Ident::Out => self.out_line_size,
        }
    }

    fn stage_line_size(&self, ident: Ident) -> u32 {
        self.smm_config.line_size(ident)
    }

    fn stage_tiling(&self, ident: Ident) -> StageTiling {
        self.smm_config.tiling(ident)
    }

    fn layout(&self, ident: Ident) -> MatrixLayout {
        match ident {
            Ident::Lhs => self.lhs_layout,
            Ident::Rhs => self.rhs_layout,
            Ident::Out => self.smm_config.layout(Ident::Out),
        }
    }

    fn num_planes(&self) -> u32 {
        self.smm_config.num_planes()
    }

    fn plane_dim(&self) -> u32 {
        self.smm_config.plane_dim()
    }

    fn tiling_layout(&self, ident: Ident) -> TilingLayout {
        self.smm_config.tiling_layout(ident)
    }

    fn check_row_bounds(&self, ident: Ident) -> bool {
        match ident {
            Ident::Lhs => self.check_m_bounds,
            Ident::Rhs => self.check_k_bounds,
            Ident::Out => self.check_m_bounds,
        }
    }

    fn check_col_bounds(&self, ident: Ident) -> bool {
        match ident {
            Ident::Lhs => self.check_k_bounds,
            Ident::Rhs => self.check_n_bounds,
            Ident::Out => self.check_n_bounds,
        }
    }

    fn transpose_load(&self, ident: Ident) -> bool {
        self.layout(ident) != self.smm_config.layout(ident)
    }

    fn load_mode(&self) -> LoadMode {
        self.load_mode
    }
}

impl<S: stage::StageConfig> MatmulConfig for Config<S> {}

impl<S: stage::StageConfig> Config<S> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        smm_config: S,
        check_m_bounds: bool,
        check_n_bounds: bool,
        check_k_bounds: bool,
        lhs_layout: MatrixLayout,
        rhs_layout: MatrixLayout,
        lhs_line_size: u32,
        rhs_line_size: u32,
        out_line_size: u32,
        k_step: u32,
        load_mode: LoadMode,
    ) -> Self {
        Self {
            smm_config,
            check_m_bounds,
            check_n_bounds,
            check_k_bounds,
            lhs_layout,
            rhs_layout,
            lhs_line_size,
            rhs_line_size,
            out_line_size,
            k_step,
            load_mode,
        }
    }
}
