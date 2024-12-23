use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::matmul::components::{
    tile::{TileConfig, TileMatmulFamily},
    Ident, InputIdent, LhsStageDim, MatmulConfig, MatmulSize, MatrixLayout, OutStageDim,
    RhsStageDim, StageDim,
};

use super::{Config, TilingOrderConfig};

pub struct CommonStageInput<TMM: TileMatmulFamily> {
    pub tile: TMM::Input,
    pub num_stages: MatmulSize,
}

pub(super) fn stage_matmul_size<TMM: TileMatmulFamily>(
    config: &TMM::Config,
    num_stage: &MatmulSize,
) -> MatmulSize {
    let size = TMM::size(config);

    MatmulSize {
        m: num_stage.m * size.m,
        n: num_stage.n * size.n,
        k: num_stage.k * size.k,
    }
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for the single buffer matmul
pub struct CommonStageConfig<T: TileConfig> {
    pub tmm_config: T,
    pub num_stage: MatmulSize,
    pub lhs_stage_dim: LhsStageDim,
    pub rhs_stage_dim: RhsStageDim,
    pub out_stage_dim: OutStageDim,
    pub num_planes: u32,
    pub lhs_tiling_order: TilingOrderConfig,
    pub rhs_tiling_order: TilingOrderConfig,
}

impl<T: TileConfig> Config for CommonStageConfig<T> {
    type TmmConfig = T;

    fn to_tmm_config(self) -> Self::TmmConfig {
        self.tmm_config
    }

    fn line_size(&self, ident: Ident) -> u32 {
        self.tmm_config.line_size(ident)
    }

    fn stage_dim(&self, ident: Ident) -> Box<dyn StageDim> {
        match ident {
            Ident::Lhs => Box::new(self.lhs_stage_dim),
            Ident::Rhs => Box::new(self.rhs_stage_dim),
            Ident::Out => Box::new(self.out_stage_dim),
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

    fn tiling_order(&self, ident: Ident) -> TilingOrderConfig {
        match ident.as_input() {
            InputIdent::Lhs => self.lhs_tiling_order,
            InputIdent::Rhs => self.rhs_tiling_order,
        }
    }
}

impl<T: TileConfig> MatmulConfig for CommonStageConfig<T> {}

impl<T: TileConfig> CommonStageConfig<T> {
    pub fn new(
        tmm_config: T,
        num_stage: MatmulSize,
        lhs_stage_dim: LhsStageDim,
        rhs_stage_dim: RhsStageDim,
        out_stage_dim: OutStageDim,
        num_planes: u32,
        lhs_tiling_order: TilingOrderConfig,
        rhs_tiling_order: TilingOrderConfig,
    ) -> Self {
        Self {
            num_stage,
            tmm_config,
            lhs_stage_dim,
            rhs_stage_dim,
            out_stage_dim,
            num_planes,
            lhs_tiling_order,
            rhs_tiling_order,
        }
    }
}
