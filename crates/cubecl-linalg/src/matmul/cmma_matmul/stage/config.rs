use crate::matmul::config::MatmulConfig;
use crate::matmul::matmul_stage::SmmConfig;
use crate::matmul::matmul_tile::TmmConfig;
use crate::matmul::matrix::{Ident, MatrixLayout};
use crate::matmul::stage_dim::StageDim;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::TilingOrderConfig;

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for the PlaneRowStageMatmul
pub struct PlaneRowStageMatmulConfig<T: TmmConfig> {
    tmm_config: T,
    lhs_stage_dim: StageDim,
    rhs_stage_dim: StageDim,
    out_stage_dim: StageDim,
    num_planes: u32,
    tiling_order: TilingOrderConfig,
}

impl<T: TmmConfig> SmmConfig for PlaneRowStageMatmulConfig<T> {
    type TmmConfig = T;

    fn to_tmm_config(self) -> Self::TmmConfig {
        self.tmm_config
    }

    fn line_size(&self, ident: Ident) -> u32 {
        self.tmm_config.line_size(ident)
    }

    fn stage_dim(&self, ident: Ident) -> StageDim {
        match ident {
            Ident::Lhs => self.lhs_stage_dim,
            Ident::Rhs => self.rhs_stage_dim,
            Ident::Out => self.out_stage_dim,
        }
    }

    fn layout(&self, ident: Ident) -> MatrixLayout {
        self.tmm_config.layout(ident)
    }

    fn num_planes(&self) -> u32 {
        self.num_planes
    }

    fn plane_dim(&self) -> u32 {
        self.tmm_config.plane_dim()
    }

    fn tiling_order(&self) -> TilingOrderConfig {
        self.tiling_order
    }
}

impl<T: TmmConfig> MatmulConfig for PlaneRowStageMatmulConfig<T> {}

impl<T: TmmConfig> PlaneRowStageMatmulConfig<T> {
    pub fn new(
        tmm_config: T,
        lhs_stage_dim: StageDim,
        rhs_stage_dim: StageDim,
        out_stage_dim: StageDim,
        num_planes: u32,
        tiling_order: TilingOrderConfig,
    ) -> Self {
        Self {
            tmm_config,
            lhs_stage_dim,
            rhs_stage_dim,
            out_stage_dim,
            num_planes,
            tiling_order,
        }
    }
}
