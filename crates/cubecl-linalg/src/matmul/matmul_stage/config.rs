use crate::matmul::cmma_matmul::stage::TilingOrderConfig;
use crate::matmul::config::{ComptimeConfig, MatmulConfig};
use crate::matmul::matmul_tile::TmmConfig;
use crate::matmul::matrix::{Ident, MatrixLayout};
use crate::matmul::stage_dim::StageDim;

pub trait SmmConfig: ComptimeConfig + MatmulConfig {
    type TmmConfig: TmmConfig;

    fn into_tmm_config(self) -> Self::TmmConfig;

    fn line_size(&self, ident: Ident) -> u32;
    fn stage_dim(&self, ident: Ident) -> StageDim;
    fn layout(&self, ident: Ident) -> MatrixLayout;
    fn num_planes(&self) -> u32;
    fn tiling_order(&self) -> TilingOrderConfig;
}
