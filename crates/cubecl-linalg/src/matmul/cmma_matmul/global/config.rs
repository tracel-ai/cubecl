use crate::matmul::cmma_matmul::stage::TilingOrderConfig;
use crate::matmul::config::{ComptimeConfig, MatmulConfig, MatmulLaunchConfig};
use crate::matmul::matmul_global::GmmConfig;
use crate::matmul::matmul_stage::SmmConfig;
use crate::matmul::matrix::{Ident, MatrixLayout};
use crate::matmul::stage_dim::StageDim;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct CmmaGlobalMatmulConfig<S: SmmConfig> {
    smm_config: S,
    out_smem_line_size: u32,
    plane_dim: u32,
    check_m_bounds: bool,
    check_n_bounds: bool,
}

impl<S: SmmConfig> ComptimeConfig for CmmaGlobalMatmulConfig<S> {}

impl<S: SmmConfig> MatmulLaunchConfig for CmmaGlobalMatmulConfig<S> {
    fn cube_dim(&self) -> CubeDim {
        CubeDim {
            x: self.plane_dim(),
            y: self.num_planes(),
            z: 1,
        }
    }

    fn cube_count(&self) -> CubeCount {
        CubeCount::Static(1, 1, 1)
    }
}

impl<S: SmmConfig> GmmConfig for CmmaGlobalMatmulConfig<S> {
    type SmmConfig = S;

    fn to_smm_config(&self) -> Self::SmmConfig {
        self.smm_config
    }

    fn line_size(&self, ident: Ident) -> u32 {
        self.smm_config.line_size(ident)
    }

    fn stage_dim(&self, ident: Ident) -> StageDim {
        self.smm_config.stage_dim(ident)
    }

    fn layout(&self, ident: Ident) -> MatrixLayout {
        self.smm_config.layout(ident)
    }

    fn out_smem_line_size(&self) -> u32 {
        self.out_smem_line_size
    }

    fn num_planes(&self) -> u32 {
        self.smm_config.num_planes()
    }

    fn plane_dim(&self) -> u32 {
        self.plane_dim
    }

    fn tiling_order(&self) -> TilingOrderConfig {
        self.smm_config.tiling_order()
    }

    fn check_m_bounds(&self) -> bool {
        self.check_m_bounds
    }

    fn check_n_bounds(&self) -> bool {
        self.check_n_bounds
    }
}

impl<S: SmmConfig> MatmulConfig for CmmaGlobalMatmulConfig<S> {}

impl<S: SmmConfig> CmmaGlobalMatmulConfig<S> {
    pub fn new(
        smm_config: S,
        out_smem_line_size: u32,
        plane_dim: u32,
        check_m_bounds: bool,
        check_n_bounds: bool,
    ) -> Self {
        Self {
            smm_config,
            out_smem_line_size,
            plane_dim,
            check_m_bounds,
            check_n_bounds,
        }
    }
}
