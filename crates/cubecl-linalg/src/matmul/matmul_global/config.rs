use crate::matmul::cmma_matmul::config::StageDim;
use crate::matmul::config::{ComptimeConfig, MatmulConfig, MatmulLaunchConfig};
use crate::matmul::matmul_stage::SmmConfig;
use crate::matmul::matrix::{Ident, MatrixLayout};

pub trait GmmConfig: ComptimeConfig + MatmulConfig + MatmulLaunchConfig {
    type SmmConfig: SmmConfig;

    fn to_smm_config(&self) -> Self::SmmConfig;

    fn line_size(&self, ident: Ident) -> u32;
    fn stage_dim(&self, ident: Ident) -> StageDim;
    fn layout(&self, ident: Ident) -> MatrixLayout;
    fn out_smem_line_size(&self) -> u32;
    fn num_planes(&self) -> u32;
    fn plane_dim(&self) -> u32;
}

pub trait ViewConfig: ComptimeConfig {}
