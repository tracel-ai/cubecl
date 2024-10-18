use crate::matmul::cmma_matmul::config::StageDim;
use crate::matmul::config::{ComptimeConfig, MatmulConfig};
use crate::matmul::matmul_tile::TmmConfig;
use crate::matmul::matrix::{Ident, MatrixLayout};

pub trait SmmConfig: ComptimeConfig + MatmulConfig {
    type TmmConfig: TmmConfig;

    fn into_tmm_config(self) -> Self::TmmConfig;

    fn line_size(&self, ident: Ident) -> u32;
    fn stage_dim(&self, ident: Ident) -> StageDim;
    fn layout(&self, ident: Ident) -> MatrixLayout;
    fn num_planes(&self) -> u32;
}
