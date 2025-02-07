use crate::matmul::components::{
    tile::{TileConfig, TileMatmulFamily},
    Ident, InputIdent, MatmulConfig, MatmulSize, MatrixLayout, StageDim,
};

use super::{StageConfig, TilingOrderConfig};

pub struct CommonStageInput {
    pub tile_shape: MatmulSize,
    pub tile_count: MatmulSize,
}

pub(super) fn stage_matmul_size<TMM: TileMatmulFamily>(
    config: &TMM::Config,
    num_stage: &MatmulSize,
) -> MatmulSize {
    let size = TMM::tile_shape(config);

    MatmulSize {
        m: num_stage.m * size.m,
        n: num_stage.n * size.n,
        k: num_stage.k * size.k,
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for the single buffer matmul
pub struct CommonStageConfig<T: TileConfig> {
    pub tmm_config: T,
    pub tile_count: MatmulSize,
    pub lhs_stage_dim: StageDim,
    pub rhs_stage_dim: StageDim,
    pub out_stage_dim: StageDim,
    pub num_planes: u32,
    pub lhs_tiling_order: TilingOrderConfig,
    pub rhs_tiling_order: TilingOrderConfig,
}

impl<T: TileConfig> StageConfig for CommonStageConfig<T> {
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

    fn tiling_order(&self, ident: Ident) -> TilingOrderConfig {
        match ident.as_input() {
            InputIdent::Lhs => self.lhs_tiling_order,
            InputIdent::Rhs => self.rhs_tiling_order,
        }
    }

    fn tile_count(&self) -> &MatmulSize {
        &self.tile_count
    }
}

impl<T: TileConfig> MatmulConfig for CommonStageConfig<T> {}

impl<T: TileConfig> CommonStageConfig<T> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tmm_config: T,
        num_stage: MatmulSize,
        lhs_stage_dim: StageDim,
        rhs_stage_dim: StageDim,
        out_stage_dim: StageDim,
        num_planes: u32,
        lhs_tiling_order: TilingOrderConfig,
        rhs_tiling_order: TilingOrderConfig,
    ) -> Self {
        Self {
            tile_count: num_stage,
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
